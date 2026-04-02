"""
dataset.py — PyTorch Dataset for DICOM chest X-ray images.
Similar data avaiable on kaggle https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?select=chest_xray

Handles:
- DICOM loading with VOI-LUT and monochrome correction (matching original pipeline)
- Grayscale → 3-channel replication (timm/ViT models expect RGB)
- Resize to IMG_SIZE × IMG_SIZE
- Patient metadata extraction (age, sex) for CatBoost
- Train / val / test splits via a simple factory function
 - torchvision augmentation pipelines for train vs eval
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold

from config import (
    TRAIN_DIR, TEST_DIR, DATA_CSV,
    IMG_SIZE, BATCH_SIZE, NUM_WORKERS, SEED, N_FOLDS,
)


# DICOM reading 

def read_dicom(path: str,
               voi_lut: bool = True,
               fix_monochrome: bool = True) -> np.ndarray:
    """
    Load a DICOM file and return a float32 numpy array in [0, 1].
    Steps
    1. Read file with pydicom.
    2. Apply VOI-LUT if available (produces perceptually correct pixel values).
    3. Fix MONOCHROME1 inversion (darker = higher value → invert so brighter = higher).
    4. Normalise to [0, 1].
    """
    dicom = pydicom.dcmread(path)

    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array.astype(np.float32)

    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data.astype(np.float32)
    data -= data.min()
    max_val = data.max()
    if max_val > 0:
        data /= max_val

    return data          # shape: (H, W), dtype float32, range [0, 1]


def extract_metadata(dicom_path: str) -> dict:
    """
    Extract patient-level metadata from a DICOM file.
    Returns a dict with keys: patient_age (int or NaN), patient_sex (str).
    """
    dicom = pydicom.dcmread(dicom_path)

    #  Age 
    age_raw = getattr(dicom, "PatientAge", "")
    try:
        # Format is typically '045Y'; strip leading zeros and 'Y'
        age = int(str(age_raw).strip().rstrip("Y").lstrip("0") or "0")
    except (ValueError, AttributeError):
        age = float("nan")

    #  Sex 
    sex = str(getattr(dicom, "PatientSex", "")).strip().upper()
    if sex not in ("M", "F"):
        sex = "O"          # Other / unknown

    return {"patient_age": age, "patient_sex": sex}


#  torchvision transform factories 

# ImageNet statistics are standard even for medical images fine-tuned from
# ImageNet pretrained weights.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        # Random contrast / brightness variation — medically plausible
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


#  Dataset ────────────────────────────────────────────────────────────────

class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for DICOM chest X-ray binary classification.

    Parameters"
    df : DataFrame with columns 'id' and optionally 'Finding'
    data_dir: directory containing .dicom files
    transform: torchvision transform to apply
    has_labels: True for train/val; False for test set
    cache: if True, cache loaded pixel arrays in memory after first load
                  (saves repeated DICOM decode; use only if RAM allows)
    """

    def __init__(self,
                 df: pd.DataFrame,
                 data_dir: str,
                 transform=None,
                 has_labels: bool = True,
                 cache: bool = False):
        self.df         = df.reset_index(drop=True)
        self.data_dir   = data_dir
        self.transform  = transform
        self.has_labels = has_labels
        self.cache      = cache
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _load_pixel_array(self, idx: int) -> np.ndarray:
        """Load and optionally cache the pixel array for index idx."""
        if self.cache and idx in self._cache:
            return self._cache[idx]

        row  = self.df.iloc[idx]
        path = os.path.join(self.data_dir, f"{row['id']}.dicom")
        arr  = read_dicom(path)

        if self.cache:
            self._cache[idx] = arr
        return arr

    def __getitem__(self, idx: int) -> dict:
        arr = self._load_pixel_array(idx)    # (H, W) float32 in [0,1]

        # Convert to uint8 PIL image, then replicate to 3 channels
        img_uint8 = (arr * 255).astype(np.uint8)
        pil_img   = Image.fromarray(img_uint8, mode="L").convert("RGB")

        if self.transform:
            tensor = self.transform(pil_img)
        else:
            tensor = transforms.ToTensor()(pil_img)

        item = {"image": tensor, "id": str(self.df.iloc[idx]["id"])}

        if self.has_labels:
            label = int(self.df.iloc[idx]["Finding"])
            item["label"] = torch.tensor(label, dtype=torch.float32)

        return item


#  Metadata DataFrame builder 

def build_metadata_df(ids: list[str], data_dir: str) -> pd.DataFrame:
    """
    Extract age and sex metadata from DICOM files for a list of image IDs.
    Returns a DataFrame indexed by 'id' with columns: patient_age, patient_sex.
    """
    records = []
    for sid in ids:
        path = os.path.join(data_dir, f"{sid}.dicom")
        meta = extract_metadata(path)
        meta["id"] = sid
        records.append(meta)
    df = pd.DataFrame(records).set_index("id")

    # Encode sex as numeric: M=0, F=1, O=2
    sex_map = {"M": 0, "F": 1, "O": 2}
    df["patient_sex"] = df["patient_sex"].map(sex_map).fillna(2).astype(int)

    # Fill missing ages with median
    median_age = df["patient_age"].median()
    df["patient_age"] = df["patient_age"].fillna(median_age)

    return df


# DataLoader factories 

def make_loaders(train_df: pd.DataFrame,
                 val_df: pd.DataFrame,
                 data_dir: str = TRAIN_DIR,
                 batch_size: int = BATCH_SIZE,
                 num_workers: int = NUM_WORKERS,
                 cache: bool = False
                 ) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.
    Returns - train_loader, val_loader
    """
    train_ds = ChestXrayDataset(
        train_df, data_dir,
        transform=get_train_transforms(),
        has_labels=True,
        cache=cache,
    )
    val_ds = ChestXrayDataset(
        val_df, data_dir,
        transform=get_eval_transforms(),
        has_labels=True,
        cache=cache,
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
                     data_dir: str = TEST_DIR,
                     batch_size: int = BATCH_SIZE,
                     num_workers: int = NUM_WORKERS) -> DataLoader:
    """Build a DataLoader for the held-out test set (no labels)."""
    test_ds = ChestXrayDataset(
        test_df, data_dir,
        transform=get_eval_transforms(),
        has_labels=False,
    )
    return DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


# Split helpers

def load_training_df() -> pd.DataFrame:
    """Load data.csv and return only rows that have a Finding label."""
    df = pd.read_csv(DATA_CSV, dtype={"id": "str", "Finding": "Int64"})
    return df.loc[~df["Finding"].isna()].copy().reset_index(drop=True)


def train_val_split(df: pd.DataFrame,
                    val_size: int = 2000,
                    seed: int = SEED
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train / validation split matching the original notebook."""
    train, val = train_test_split(
        df, test_size=val_size,
        random_state=seed, stratify=df["Finding"],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True)


def get_kfold_splits(df: pd.DataFrame,
                     n_folds: int = N_FOLDS,
                     seed: int = SEED) -> list[tuple]:
    """
    Return a list of (train_df, val_df) tuples for stratified k-fold CV.
    Used during OOF generation to train CatBoost without leakage.
    """
    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in skf.split(df, df["Finding"]):
        splits.append((
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True),
        ))
    return splits


def load_test_df() -> pd.DataFrame:
    """Build a DataFrame of test image IDs (no labels)."""
    import pathlib
    paths = list(pathlib.Path(TEST_DIR).glob("*.dicom"))
    ids   = [p.stem for p in paths]
    return pd.DataFrame({"id": ids, "Finding": pd.NA})


# Positive weight for weighted BCE loss 

def compute_pos_weight(df: pd.DataFrame) -> torch.Tensor:
    """
    Compute pos_weight = n_negative / n_positive for BCEWithLogitsLoss.
    Returned as a 1-element tensor.
    """
    counts     = df["Finding"].value_counts()
    n_neg      = float(counts.get(0, 1))
    n_pos      = float(counts.get(1, 1))
    pos_weight = n_neg / n_pos
    print(f"  pos_weight = {pos_weight:.3f}  "
          f"(n_neg={int(n_neg)}, n_pos={int(n_pos)})")
    return torch.tensor([pos_weight], dtype=torch.float32)
