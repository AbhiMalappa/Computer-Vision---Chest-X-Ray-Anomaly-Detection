"""
dataset.py — PyTorch Dataset for NIH ChestX-ray14 (Paper 1).

Key differences from VinBigData pipeline:
  - Images are PNG (1024×1024), not DICOM
  - Labels come from Data_Entry_2017.csv, not separate label file
  - Binary mapping: "No Finding" → 0, any disease → 1
  - Metadata (age, sex) extracted directly from CSV — no DICOM parsing needed
  - Split MUST be patient-level to prevent leakage across follow-up images
    (same patient can appear in multiple images; naive image-level split
     would put the same patient in both train and test)
  - 3-fold OOF (reduced from 5 for compute efficiency on 112k images)
"""

import os
import glob as _glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold

from config import (
    DATA_DIR, IMAGES_DIR, DATA_ENTRY_CSV,
    IMG_SIZE, BATCH_SIZE, NUM_WORKERS,
    SEED, N_FOLDS,
    NO_FINDING_LABEL,
    VAL_PATIENT_FRAC, TEST_PATIENT_FRAC,
    DEBUG_SUBSET,
)


# ─── Image index ──────────────────────────────────────────────────────────────
# Builds a mapping of {image_id: full_path} that handles both:
#   - Local flat structure:  ./data/images/00000001_000.png
#   - Kaggle split structure: .../images_001/00000001_000.png  ... images_012/
def _build_image_index() -> dict:
    index = {}
    # Flat images/ directory (local)
    if os.path.isdir(IMAGES_DIR):
        for fname in os.listdir(IMAGES_DIR):
            if fname.endswith(".png"):
                index[fname] = os.path.join(IMAGES_DIR, fname)
    # Split images_001 … images_012 subdirectories (Kaggle)
    for subdir in sorted(_glob.glob(os.path.join(DATA_DIR, "images_*"))):
        if os.path.isdir(subdir):
            for fname in os.listdir(subdir):
                if fname.endswith(".png"):
                    index[fname] = os.path.join(subdir, fname)
    return index

IMAGE_INDEX = _build_image_index()


# ─── torchvision transforms ───────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── Label CSV loader ─────────────────────────────────────────────────────────

def load_nih_csv(csv_path: str = DATA_ENTRY_CSV) -> pd.DataFrame:
    """
    Load Data_Entry_2017.csv and return a clean DataFrame with:
      - image_id      : filename (e.g. '00000001_000.png')
      - patient_id    : integer patient identifier
      - finding_labels: raw pipe-separated string (e.g. 'Atelectasis|Effusion')
      - binary_label  : 0 = No Finding, 1 = any disease present
      - patient_age   : integer age
      - patient_sex   : encoded integer (M=0, F=1, other=2)

    Data_Entry_2017.csv columns:
      Image Index, Finding Labels, Follow-up #,
      Patient ID, Patient Age, Patient Gender,
      View Position, OriginalImage[Width Height],
      OriginalImagePixelSpacing[x y]
    """
    df = pd.read_csv(csv_path)

    # Standardise column names
    df = df.rename(columns={
        "Image Index":     "image_id",
        "Finding Labels":  "finding_labels",
        "Patient ID":      "patient_id",
        "Patient Age":     "patient_age",
        "Patient Gender":  "patient_gender",
    })

    # Binary label
    df["binary_label"] = df["finding_labels"].apply(
        lambda x: 0 if str(x).strip() == NO_FINDING_LABEL else 1
    )

    # Encode sex
    sex_map = {"M": 0, "F": 1}
    df["patient_sex"] = df["patient_gender"].map(sex_map).fillna(2).astype(int)

    # Clean age — occasionally has string artifacts
    df["patient_age"] = pd.to_numeric(df["patient_age"], errors="coerce")
    median_age = df["patient_age"].median()
    df["patient_age"] = df["patient_age"].fillna(median_age).astype(float)

    # Filter to only images that exist on disk (handles partial downloads)
    before = len(df)
    df = df[df["image_id"].isin(IMAGE_INDEX)].reset_index(drop=True)
    if len(df) < before:
        print(f"  Filtered to images on disk: {len(df):,} / {before:,}")

    print(f"  Loaded {len(df):,} images from {df['patient_id'].nunique():,} patients")
    neg = (df["binary_label"] == 0).sum()
    pos = (df["binary_label"] == 1).sum()
    print(f"  Binary label: neg={neg:,} ({neg/len(df)*100:.1f}%)  "
          f"pos={pos:,} ({pos/len(df)*100:.1f}%)")

    return df[["image_id", "patient_id", "finding_labels",
               "binary_label", "patient_age", "patient_sex"]]


# ─── Patient-level split ──────────────────────────────────────────────────────

def patient_level_split(df: pd.DataFrame,
                        val_frac: float = VAL_PATIENT_FRAC,
                        test_frac: float = TEST_PATIENT_FRAC,
                        seed: int = SEED
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data at the PATIENT level to prevent leakage.

    Each patient has multiple images (follow-up scans). A naive image-level
    split would allow the same patient's anatomy to appear in both train and
    test, inflating performance metrics.

    Strategy
    --------
    1. Build a patient-level DataFrame with one row per patient.
    2. Assign each patient a majority-vote binary label
       (if any image for that patient is positive, patient = positive).
    3. Stratified split patients into train / val / test.
    4. Map patient assignments back to the image-level DataFrame.

    Returns
    -------
    train_df, val_df, test_df  — image-level DataFrames
    """
    # Patient-level label: positive if ANY image is positive
    patient_df = (
        df.groupby("patient_id")["binary_label"]
        .max()
        .reset_index()
        .rename(columns={"binary_label": "patient_label"})
    )

    # Debug mode: subsample to a small number of patients for fast testing
    if DEBUG_SUBSET is not None:
        n_sample = min(DEBUG_SUBSET, len(patient_df))
        patient_df = patient_df.sample(n=n_sample, random_state=seed).reset_index(drop=True)
        df = df[df["patient_id"].isin(patient_df["patient_id"])].reset_index(drop=True)
        print(f"  [DEBUG] Subsampled to {n_sample} patients / {len(df)} images")

    n_patients = len(patient_df)
    n_val      = int(n_patients * val_frac)
    n_test     = int(n_patients * test_frac)

    # First split off test patients
    trainval_patients, test_patients = train_test_split(
        patient_df,
        test_size=n_test,
        stratify=patient_df["patient_label"],
        random_state=seed,
    )

    # Then split trainval into train / val
    train_patients, val_patients = train_test_split(
        trainval_patients,
        test_size=n_val,
        stratify=trainval_patients["patient_label"],
        random_state=seed,
    )

    train_ids = set(train_patients["patient_id"])
    val_ids   = set(val_patients["patient_id"])
    test_ids  = set(test_patients["patient_id"])

    train_df = df[df["patient_id"].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df["patient_id"].isin(val_ids)].reset_index(drop=True)
    test_df  = df[df["patient_id"].isin(test_ids)].reset_index(drop=True)

    print(f"\n  Patient-level split:")
    print(f"    Train : {len(train_patients):5,} patients  →  {len(train_df):6,} images  "
          f"(pos={train_df['binary_label'].mean()*100:.1f}%)")
    print(f"    Val   : {len(val_patients):5,} patients  →  {len(val_df):6,} images  "
          f"(pos={val_df['binary_label'].mean()*100:.1f}%)")
    print(f"    Test  : {len(test_patients):5,} patients  →  {len(test_df):6,} images  "
          f"(pos={test_df['binary_label'].mean()*100:.1f}%)")

    return train_df, val_df, test_df


def get_kfold_splits(df: pd.DataFrame,
                     n_folds: int = N_FOLDS,
                     seed: int = SEED) -> list[tuple]:
    """
    Patient-level stratified K-fold splits for OOF generation.

    Returns list of (train_df, val_df) image-level DataFrames.
    Patient leakage is prevented — no patient appears in both
    train and val folds.
    """
    # Build patient-level labels
    patient_df = (
        df.groupby("patient_id")["binary_label"]
        .max()
        .reset_index()
    )

    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=seed)
    splits = []

    for train_pat_idx, val_pat_idx in skf.split(
        patient_df["patient_id"],
        patient_df["binary_label"]
    ):
        train_patient_ids = set(
            patient_df.iloc[train_pat_idx]["patient_id"]
        )
        val_patient_ids = set(
            patient_df.iloc[val_pat_idx]["patient_id"]
        )

        # Do NOT reset_index here — original indices are needed by
        # generate_oof.py to map val predictions back to the right
        # positions in the oof_probs array.
        fold_train = df[df["patient_id"].isin(train_patient_ids)]
        fold_val   = df[df["patient_id"].isin(val_patient_ids)]

        splits.append((fold_train, fold_val))

    return splits


# ─── NIH ChestX-ray14 Dataset ────────────────────────────────────────────────

class NIHChestXrayDataset(Dataset):
    """
    PyTorch Dataset for NIH ChestX-ray14 PNG images.

    Parameters
    ----------
    df          : DataFrame with columns from load_nih_csv()
                  Must contain: image_id, binary_label (if has_labels=True),
                  patient_age, patient_sex
    images_dir  : directory containing PNG image files
    transform   : torchvision transform pipeline
    has_labels  : True for train/val; False for test
    """

    def __init__(self,
                 df: pd.DataFrame,
                 images_dir: str = IMAGES_DIR,
                 transform=None,
                 has_labels: bool = True):
        self.df         = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform  = transform
        self.has_labels = has_labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row      = self.df.iloc[idx]
        img_path = IMAGE_INDEX.get(
            row["image_id"],
            os.path.join(self.images_dir, row["image_id"])  # fallback
        )

        # Load PNG — already 8-bit grayscale, convert to RGB for timm/ViT
        pil_img = Image.open(img_path).convert("RGB")

        if self.transform:
            tensor = self.transform(pil_img)
        else:
            tensor = transforms.ToTensor()(pil_img)

        item = {"image": tensor, "id": str(row["image_id"])}

        if self.has_labels:
            item["label"] = torch.tensor(
                int(row["binary_label"]), dtype=torch.float32
            )

        return item


# ─── DataLoader factories ────────────────────────────────────────────────────

def make_loaders(train_df: pd.DataFrame,
                 val_df: pd.DataFrame,
                 images_dir: str = IMAGES_DIR,
                 batch_size: int = BATCH_SIZE,
                 num_workers: int = NUM_WORKERS
                 ) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders for NIH ChestX-ray14."""

    train_ds = NIHChestXrayDataset(
        train_df, images_dir,
        transform=get_train_transforms(),
        has_labels=True,
    )
    val_ds = NIHChestXrayDataset(
        val_df, images_dir,
        transform=get_eval_transforms(),
        has_labels=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def make_test_loader(test_df: pd.DataFrame,
                     images_dir: str = IMAGES_DIR,
                     batch_size: int = BATCH_SIZE,
                     num_workers: int = NUM_WORKERS) -> DataLoader:
    """Build DataLoader for the held-out test set."""
    test_ds = NIHChestXrayDataset(
        test_df, images_dir,
        transform=get_eval_transforms(),
        has_labels=True,    # test_df has labels for evaluation
    )
    return DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


# ─── Metadata builder for CatBoost ───────────────────────────────────────────

def build_metadata_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract age and sex from the NIH CSV DataFrame for CatBoost features.
    Much simpler than DICOM extraction — metadata is already in the CSV.

    Returns DataFrame indexed by image_id with columns:
      patient_age (float), patient_sex (int: M=0, F=1, other=2)
    """
    meta = df[["image_id", "patient_age", "patient_sex"]].copy()
    meta = meta.set_index("image_id")
    return meta


# ─── Positive weight for BCE loss ────────────────────────────────────────────

def compute_pos_weight(df: pd.DataFrame) -> torch.Tensor:
    """
    Compute pos_weight = n_negative / n_positive for BCEWithLogitsLoss.
    """
    counts     = df["binary_label"].value_counts()
    n_neg      = float(counts.get(0, 1))
    n_pos      = float(counts.get(1, 1))
    pos_weight = n_neg / n_pos
    print(f"  pos_weight = {pos_weight:.3f}  "
          f"(n_neg={int(n_neg):,}, n_pos={int(n_pos):,})")
    return torch.tensor([pos_weight], dtype=torch.float32)
