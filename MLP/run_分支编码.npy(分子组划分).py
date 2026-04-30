# =====================================================
# Branch Encoding MLP (GroupSplit by InChIKey) - FULL METRICS VERSION
# =====================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# ===============================
# 1. 配置区
# ===============================

DATA_ROOT = r"D:\实验\数据\Nist数据\MLP\1008403训练MLP模型"

FP_NPY     = os.path.join(DATA_ROOT, "fp.npy")
MS_NPY     = os.path.join(DATA_ROOT, "ms.npy")
Y_NPY      = os.path.join(DATA_ROOT, "y.npy")
INCHI_NPY  = os.path.join(DATA_ROOT, "inchikey.npy")

BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 100
RANDOM_STATE = 42
COSINE_WEIGHT = 0.1
EARLY_STOP_PATIENCE = 10

assert torch.cuda.is_available(), "CUDA 不可用"
DEVICE = torch.device("cuda")

LOG_DIR = os.path.join(DATA_ROOT, "branch_mlp_group_split_metrics")
os.makedirs(LOG_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(LOG_DIR, "best_model.pt")

# ===============================
# 2. Dataset
# ===============================

class BranchDataset(Dataset):
    def __init__(self, fp, ms, y, indices):
        self.fp = fp
        self.ms = ms
        self.y  = y
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return self.fp[i], self.ms[i], self.y[i]

# ===============================
# 3. 模型
# ===============================

class BranchMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fp_branch = nn.Sequential(
            nn.Linear(1002, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )

        self.ms_branch = nn.Sequential(
            nn.Linear(22, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
        )

    def forward(self, fp, ms):
        x_fp = self.fp_branch(fp)
        x_ms = self.ms_branch(ms)
        x = torch.cat([x_fp, x_ms], dim=1)
        return self.fusion(x)

# ===============================
# 4. Loss
# ===============================

def mse_cosine_loss(pred, target, w):
    mse = F.mse_loss(pred, target)
    cos_sim = F.cosine_similarity(pred, target, dim=1).mean()
    cos_loss = 1 - cos_sim
    total_loss = mse + w * cos_loss
    return total_loss, mse, cos_loss, cos_sim

# ===============================
# 5. 主流程
# ===============================

def main():

    writer = SummaryWriter(LOG_DIR)

    print("Loading data into RAM...")
    fp_all = np.load(FP_NPY).astype(np.float32)
    ms_all = np.load(MS_NPY).astype(np.float32)
    y_all  = np.load(Y_NPY).astype(np.float32)
    inchikey = np.load(INCHI_NPY, allow_pickle=True)
    print("Data loaded.")

    N = fp_all.shape[0]
    all_indices = np.arange(N)

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.1,
        random_state=RANDOM_STATE
    )

    train_idx, val_idx = next(
        gss.split(all_indices, groups=inchikey)
    )

    print("===================================")
    print("Total samples:", N)
    print("Train samples:", len(train_idx))
    print("Val samples:", len(val_idx))
    print("===================================")

    fp_all = torch.from_numpy(fp_all)
    ms_all = torch.from_numpy(ms_all)
    y_all  = torch.from_numpy(y_all)

    train_loader = DataLoader(
        BranchDataset(fp_all, ms_all, y_all, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        BranchDataset(fp_all, ms_all, y_all, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    model = BranchMLP().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4
    )

    best_val = float("inf")
    patience = 0

    # ===========================
    # 训练循环
    # ===========================

    for epoch in range(1, EPOCHS + 1):

        # ===== Train =====
        model.train()

        train_total = 0
        train_mse = 0
        train_cos_loss = 0
        train_cos_sim = 0

        for fp, ms, y in tqdm(train_loader, desc=f"Train {epoch}"):

            fp = fp.to(DEVICE, non_blocking=True)
            ms = ms.to(DEVICE, non_blocking=True)
            y  = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            pred = model(fp, ms)

            total, mse, cos_loss, cos_sim = mse_cosine_loss(
                pred, y, COSINE_WEIGHT
            )

            total.backward()
            optimizer.step()

            train_total += total.item()
            train_mse += mse.item()
            train_cos_loss += cos_loss.item()
            train_cos_sim += cos_sim.item()

        train_total /= len(train_loader)
        train_mse /= len(train_loader)
        train_cos_loss /= len(train_loader)
        train_cos_sim /= len(train_loader)

        # ===== Validation =====
        model.eval()

        val_total = 0
        val_mse = 0
        val_cos_loss = 0
        val_cos_sim = 0

        with torch.no_grad():
            for fp, ms, y in val_loader:

                fp = fp.to(DEVICE, non_blocking=True)
                ms = ms.to(DEVICE, non_blocking=True)
                y  = y.to(DEVICE, non_blocking=True)

                pred = model(fp, ms)

                total, mse, cos_loss, cos_sim = mse_cosine_loss(
                    pred, y, COSINE_WEIGHT
                )

                val_total += total.item()
                val_mse += mse.item()
                val_cos_loss += cos_loss.item()
                val_cos_sim += cos_sim.item()

        val_total /= len(val_loader)
        val_mse /= len(val_loader)
        val_cos_loss /= len(val_loader)
        val_cos_sim /= len(val_loader)

        # ===== TensorBoard =====
        writer.add_scalar("train/total_loss", train_total, epoch)
        writer.add_scalar("train/mse", train_mse, epoch)
        writer.add_scalar("train/cos_loss", train_cos_loss, epoch)
        writer.add_scalar("train/cos_sim", train_cos_sim, epoch)

        writer.add_scalar("val/total_loss", val_total, epoch)
        writer.add_scalar("val/mse", val_mse, epoch)
        writer.add_scalar("val/cos_loss", val_cos_loss, epoch)
        writer.add_scalar("val/cos_sim", val_cos_sim, epoch)

        print(
            f"[Epoch {epoch:03d}] "
            f"Train Loss: {train_total:.4f} | "
            f"Val Loss: {val_total:.4f} | "
            f"Val MSE: {val_mse:.4f} | "
            f"Val CosSim: {val_cos_sim:.4f}"
        )

        # ===== 保存 best =====
        if val_total < best_val:
            best_val = val_total
            patience = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"✅ Best model saved at epoch {epoch}")
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print("⛔ Early stopping triggered.")
                break

    print("Loading best model...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()

    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    main()