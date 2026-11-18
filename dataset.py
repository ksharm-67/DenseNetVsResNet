import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "No Finding",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
#No Finding excluded from mean AUC
DISEASE_IDX = [i for i, c in enumerate(CLASSES) if c != "No Finding"]


def build_path_index(data_root):
    pattern = os.path.join(data_root, "images_*", "images", "*.png")
    index = {os.path.basename(p): p for p in glob.glob(pattern)}
    if not index:
        raise FileNotFoundError(
            f"No images found with pattern {pattern}. Check --data_root."
        )
    return index


def build_splits(data_root, val_frac=0.1, split_seed=42):
    #Split by patient, test set is the official test_list
    df = pd.read_csv(os.path.join(data_root, "Data_Entry_2017.csv"))

    def read_list(name):
        with open(os.path.join(data_root, name)) as f:
            return {line.strip() for line in f if line.strip()}

    train_val_files = read_list("train_val_list.txt")
    test_files = read_list("test_list.txt")

    class_to_col = {c: i for i, c in enumerate(CLASSES)}
    labels = np.zeros((len(df), len(CLASSES)), dtype=np.float32)
    for row, finding in enumerate(df["Finding Labels"]):
        for name in finding.split("|"):
            labels[row, class_to_col[name]] = 1.0

    names = df["Image Index"].values
    patients = df["Patient ID"].values
    tv_idx = np.where(df["Image Index"].isin(train_val_files).values)[0]
    test_idx = np.where(df["Image Index"].isin(test_files).values)[0]

    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=split_seed)
    tr, va = next(gss.split(tv_idx, groups=patients[tv_idx]))
    train_idx, val_idx = tv_idx[tr], tv_idx[va]

    path_index = build_path_index(data_root)

    def pack(ix):
        return [path_index[n] for n in names[ix]], labels[ix]

    return {"train": pack(train_idx), "val": pack(val_idx), "test": pack(test_idx)}


class CustomImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None, cache_dir=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.cache_dir = cache_dir
        self.arr = None
        if cache_dir:
            with open(os.path.join(cache_dir, "names.json")) as f:
                name_to_idx = {n: i for i, n in enumerate(json.load(f))}
            self.cache_idx = [name_to_idx[os.path.basename(p)] for p in paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.cache_dir:
            if self.arr is None:
                self.arr = np.load(os.path.join(self.cache_dir, "images.npy"), mmap_mode="r")
            image = Image.fromarray(np.array(self.arr[self.cache_idx[idx]]))
        else:
            image = Image.open(self.paths[idx]).convert("L")
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label
