# dataset.py
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapDataset(Dataset):
    """
    基于 .npy memmap 的 Dataset
    """

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

        meta_path = self.data_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"缺失 meta.json: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.input_ids = np.load(self.data_dir / "input_ids.npy", mmap_mode="r")
        self.target_ids = np.load(self.data_dir / "target_ids.npy", mmap_mode="r")
        self.token_mask = np.load(self.data_dir / "token_mask.npy", mmap_mode="r")
        self.cond = np.load(self.data_dir / "cond.npy", mmap_mode="r")
        self.exact_mass = np.load(self.data_dir / "exact_mass.npy", mmap_mode="r")

        n = self.meta["num_samples"]
        if not (
            len(self.input_ids) == len(self.target_ids) == len(self.token_mask) ==
            len(self.cond) == len(self.exact_mass) == n
        ):
            raise ValueError("memmap 文件长度不一致，请检查。")

    def __len__(self):
        return self.meta["num_samples"]

    def __getitem__(self, idx):
        return {
            "input_ids": torch.from_numpy(np.asarray(self.input_ids[idx], dtype=np.int64)),
            "target_ids": torch.from_numpy(np.asarray(self.target_ids[idx], dtype=np.int64)),
            "token_mask": torch.from_numpy(np.asarray(self.token_mask[idx], dtype=np.int64)),
            "cond_vec": torch.from_numpy(np.asarray(self.cond[idx], dtype=np.float32)),
            "exact_mass": torch.tensor(self.exact_mass[idx], dtype=torch.float32),
        }