# =====================================================
# Branch Encoding MLP (GroupSplit by InChIKey)
# FULL MEMORY LOADING VERSION - 修正版
# - 全内存加载
# - Windows 下避免 DataLoader 多进程序列化超大数组报错
# - 保留 AMP、ReduceLROnPlateau、早停、checkpoint
# =====================================================

import os
import platform
import random
import time
from multiprocessing import freeze_support

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

DATA_ROOT = r"D:\实验\数据\Nist数据\MLP\1008403"

FP_NPY = os.path.join(DATA_ROOT, "fp.npy")
MS_NPY = os.path.join(DATA_ROOT, "ms.npy")
Y_NPY = os.path.join(DATA_ROOT, "y.npy")
INCHI_NPY = os.path.join(DATA_ROOT, "inchikey.npy")

BATCH_SIZE = 256   # 可根据显存改成 128 / 256 / 512
EPOCHS = 100
RANDOM_STATE = 42

# loss 权重
MSE_WEIGHT = 1.0
COSINE_WEIGHT = 0.2

# 优化器
LR = 1e-3
WEIGHT_DECAY = 1e-4

# scheduler
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-6

# early stopping
EARLY_STOP_PATIENCE = 10
MIN_DELTA = 1e-4

# 训练稳定性
GRAD_CLIP_NORM = 5.0

# 恢复训练
RESUME_TRAINING = True

# DataLoader 配置
# Windows 下对“全内存大数组 Dataset”最稳的是 num_workers=0
if platform.system().lower() == "windows":
    NUM_WORKERS = 0
    PERSISTENT_WORKERS = False
    PREFETCH_FACTOR = None
else:
    NUM_WORKERS = 4
    PERSISTENT_WORKERS = True
    PREFETCH_FACTOR = 2

PIN_MEMORY = True

assert torch.cuda.is_available(), "CUDA 不可用"
DEVICE = torch.device("cuda")

torch.set_float32_matmul_precision("high")

LOG_DIR = os.path.join(DATA_ROOT, "branch_mlp_group_split_memory_fixed")
os.makedirs(LOG_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(LOG_DIR, "best_model.pt")
LAST_CHECKPOINT_PATH = os.path.join(LOG_DIR, "last_checkpoint.pt")

# ===============================
# 2. 随机种子
# ===============================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ===============================
# 3. Dataset
# ===============================

class MemoryDataset(Dataset):
    """
    全内存加载版 Dataset
    - 数据已经在 RAM 中
    - 这里直接按索引取
    - 返回 torch tensor，减少 DataLoader 的隐式转换
    """

    def __init__(self, indices, fp_data, ms_data, y_data):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.fp_data = fp_data
        self.ms_data = ms_data
        self.y_data = y_data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        fp = torch.from_numpy(self.fp_data[real_idx])
        ms = torch.from_numpy(self.ms_data[real_idx])
        y  = torch.from_numpy(self.y_data[real_idx])

        return fp, ms, y

# ===============================
# 4. 模型
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
# 5. Loss
# ===============================

def weighted_mse_cosine_loss(pred, target, mse_weight=1.0, cos_weight=0.1):
    mse = F.mse_loss(pred, target)
    cos_sim = F.cosine_similarity(pred, target, dim=1).mean()
    cos_loss = 1.0 - cos_sim
    total_loss = mse_weight * mse + cos_weight * cos_loss
    return total_loss, mse, cos_loss, cos_sim

# ===============================
# 6. 工具函数
# ===============================

def get_array_shape_and_dtype(path, name):
    arr = np.load(path, mmap_mode="r")
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
    return arr.shape, arr.dtype

def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler,
                    best_val, best_epoch, patience):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val": best_val,
        "best_epoch": best_epoch,
        "patience": patience,
    }
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    return (
        ckpt["epoch"],
        ckpt["best_val"],
        ckpt["best_epoch"],
        ckpt["patience"]
    )

# ===============================
# 7. 单轮训练 / 验证
# ===============================

def run_one_epoch_train(model, loader, optimizer, scaler):
    model.train()

    total_sum = 0.0
    mse_sum = 0.0
    cos_loss_sum = 0.0
    cos_sim_sum = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)

    for fp, ms, y in pbar:
        fp = fp.to(DEVICE, non_blocking=True)
        ms = ms.to(DEVICE, non_blocking=True)
        y  = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda"):
            pred = model(fp, ms)
            total, mse, cos_loss, cos_sim = weighted_mse_cosine_loss(
                pred, y,
                mse_weight=MSE_WEIGHT,
                cos_weight=COSINE_WEIGHT
            )

        scaler.scale(total).backward()

        if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        scaler.step(optimizer)
        scaler.update()

        total_sum += total.item()
        mse_sum += mse.item()
        cos_loss_sum += cos_loss.item()
        cos_sim_sum += cos_sim.item()

        pbar.set_postfix({
            "loss": f"{total.item():.4f}",
            "mse": f"{mse.item():.4f}",
            "cos": f"{cos_sim.item():.4f}"
        })

    n_batches = len(loader)
    return (
        total_sum / n_batches,
        mse_sum / n_batches,
        cos_loss_sum / n_batches,
        cos_sim_sum / n_batches
    )

@torch.no_grad()
def run_one_epoch_val(model, loader):
    model.eval()

    total_sum = 0.0
    mse_sum = 0.0
    cos_loss_sum = 0.0
    cos_sim_sum = 0.0

    pbar = tqdm(loader, desc="Validation", leave=False)

    for fp, ms, y in pbar:
        fp = fp.to(DEVICE, non_blocking=True)
        ms = ms.to(DEVICE, non_blocking=True)
        y  = y.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(device_type="cuda"):
            pred = model(fp, ms)
            total, mse, cos_loss, cos_sim = weighted_mse_cosine_loss(
                pred, y,
                mse_weight=MSE_WEIGHT,
                cos_weight=COSINE_WEIGHT
            )

        total_sum += total.item()
        mse_sum += mse.item()
        cos_loss_sum += cos_loss.item()
        cos_sim_sum += cos_sim.item()

    n_batches = len(loader)
    return (
        total_sum / n_batches,
        mse_sum / n_batches,
        cos_loss_sum / n_batches,
        cos_sim_sum / n_batches
    )

# ===============================
# 8. 主流程
# ===============================

def main():
    seed_everything(RANDOM_STATE)

    writer = SummaryWriter(LOG_DIR)

    print("===================================")
    print("Step 1/6: Checking file shapes/dtypes")
    print("===================================")
    fp_shape, fp_dtype = get_array_shape_and_dtype(FP_NPY, "fp.npy")
    ms_shape, ms_dtype = get_array_shape_and_dtype(MS_NPY, "ms.npy")
    y_shape, y_dtype   = get_array_shape_and_dtype(Y_NPY, "y.npy")

    print("===================================")
    print("Step 2/6: LOADING ALL DATA INTO MEMORY")
    print("===================================")
    print("This may take a few minutes...")

    # 原始文件已经是 float32，这里不要再 astype 复制
    print("Loading fp.npy...")
    fp_all = np.load(FP_NPY)
    print("Loading ms.npy...")
    ms_all = np.load(MS_NPY)
    print("Loading y.npy...")
    y_all  = np.load(Y_NPY)
    print("Loading inchikey.npy...")
    inchikey = np.load(INCHI_NPY, allow_pickle=True)

    # 内存使用估算
    fp_mem = fp_all.nbytes / 1024 ** 3
    ms_mem = ms_all.nbytes / 1024 ** 3
    y_mem  = y_all.nbytes / 1024 ** 3
    total_mem = fp_mem + ms_mem + y_mem

    print("\nMemory usage:")
    print(f"  fp.npy: {fp_mem:.2f} GB")
    print(f"  ms.npy: {ms_mem:.2f} GB")
    print(f"  y.npy:  {y_mem:.2f} GB")
    print(f"  Total:  {total_mem:.2f} GB")

    if total_mem > 32:
        print("⚠️ Warning: Memory usage is high.")
    else:
        print("✅ Memory usage is acceptable.")

    N = fp_all.shape[0]

    assert ms_all.shape[0] == N, f"ms.npy 行数不一致: {ms_all.shape[0]} vs {N}"
    assert y_all.shape[0] == N,  f"y.npy 行数不一致: {y_all.shape[0]} vs {N}"
    assert len(inchikey) == N,   f"inchikey.npy 长度不一致: {len(inchikey)} vs {N}"

    assert fp_all.shape[1] == 1002, f"fp.npy 第二维应为1002，实际为 {fp_all.shape[1]}"
    assert ms_all.shape[1] == 22,   f"ms.npy 第二维应为22，实际为 {ms_all.shape[1]}"
    assert y_all.shape[1] == 1024,  f"y.npy 第二维应为1024，实际为 {y_all.shape[1]}"

    print("===================================")
    print("Step 3/6: Group split by InChIKey")
    print("===================================")

    all_indices = np.arange(N)

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.1,
        random_state=RANDOM_STATE
    )

    train_idx, val_idx = next(gss.split(all_indices, groups=inchikey))

    print("Total samples:", N)
    print("Train samples:", len(train_idx))
    print("Val samples:", len(val_idx))
    print("Train ratio:", len(train_idx) / N)
    print("Val ratio:", len(val_idx) / N)

    del inchikey

    print("===================================")
    print("Step 4/6: Building DataLoader")
    print("===================================")

    train_dataset = MemoryDataset(train_idx, fp_all, ms_all, y_all)
    val_dataset   = MemoryDataset(val_idx, fp_all, ms_all, y_all)

    train_loader_kwargs = dict(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader_kwargs = dict(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    if NUM_WORKERS > 0:
        train_loader_kwargs["persistent_workers"] = PERSISTENT_WORKERS
        val_loader_kwargs["persistent_workers"] = PERSISTENT_WORKERS
        train_loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR
        val_loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(**val_loader_kwargs)

    print("NUM_WORKERS:", NUM_WORKERS)
    print("PERSISTENT_WORKERS:", PERSISTENT_WORKERS)
    print("PREFETCH_FACTOR:", PREFETCH_FACTOR if NUM_WORKERS > 0 else None)
    print("PIN_MEMORY:", PIN_MEMORY)
    print("BATCH_SIZE:", BATCH_SIZE)

    print("===================================")
    print("Step 5/6: Initializing model")
    print("===================================")

    model = BranchMLP().to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=MIN_LR
    )

    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    patience = 0

    if RESUME_TRAINING and os.path.exists(LAST_CHECKPOINT_PATH):
        print("===================================")
        print("Step 6/6: Resuming from checkpoint")
        print("===================================")
        last_epoch, best_val, best_epoch, patience = load_checkpoint(
            LAST_CHECKPOINT_PATH,
            model,
            optimizer,
            scheduler,
            scaler,
            DEVICE
        )
        start_epoch = last_epoch + 1
        print(f"Resume from epoch {start_epoch}")
        print(f"best_val = {best_val:.6f}, best_epoch = {best_epoch}, patience = {patience}")
    else:
        print("===================================")
        print("Step 6/6: Starting fresh training")
        print("===================================")

    print("\n🚀 Starting training...")
    print(f"Total epochs: {EPOCHS}")
    print(f"Batches per epoch: {len(train_loader)}")
    print("=" * 50)

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.perf_counter()

        train_total, train_mse, train_cos_loss, train_cos_sim = run_one_epoch_train(
            model, train_loader, optimizer, scaler
        )

        val_total, val_mse, val_cos_loss, val_cos_sim = run_one_epoch_val(
            model, val_loader
        )

        scheduler.step(val_total)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.perf_counter() - epoch_start

        writer.add_scalar("train/total_loss", train_total, epoch)
        writer.add_scalar("train/mse", train_mse, epoch)
        writer.add_scalar("train/cos_loss", train_cos_loss, epoch)
        writer.add_scalar("train/cos_sim", train_cos_sim, epoch)
        writer.add_scalar("train/lr", current_lr, epoch)

        writer.add_scalar("val/total_loss", val_total, epoch)
        writer.add_scalar("val/mse", val_mse, epoch)
        writer.add_scalar("val/cos_loss", val_cos_loss, epoch)
        writer.add_scalar("val/cos_sim", val_cos_sim, epoch)
        writer.add_scalar("time/epoch", epoch_time, epoch)

        print(
            f"[Epoch {epoch:03d}/{EPOCHS}] "
            f"Time: {epoch_time:.1f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_total:.4f} | "
            f"Val Loss: {val_total:.4f} | "
            f"Val MSE: {val_mse:.4f} | "
            f"Val CosSim: {val_cos_sim:.4f}"
        )

        save_checkpoint(
            LAST_CHECKPOINT_PATH,
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            best_val,
            best_epoch,
            patience
        )

        if val_total < best_val - MIN_DELTA:
            best_val = val_total
            best_epoch = epoch
            patience = 0

            torch.save(model.state_dict(), BEST_MODEL_PATH)

            save_checkpoint(
                LAST_CHECKPOINT_PATH,
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                best_val,
                best_epoch,
                patience
            )

            print(f"✅ Best model saved at epoch {epoch} | best_val = {best_val:.6f}")
        else:
            patience += 1
            print(f"⏳ No improvement. patience = {patience}/{EARLY_STOP_PATIENCE}")

            if patience >= EARLY_STOP_PATIENCE:
                print("⛔ Early stopping triggered.")
                break

    print(f"\nLoading best model from epoch {best_epoch} ...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("\nFinal validation with best model:")
    final_val_total, final_val_mse, final_val_cos_loss, final_val_cos_sim = run_one_epoch_val(
        model, val_loader
    )
    print(f"Final Val Loss: {final_val_total:.6f}")
    print(f"Final Val MSE: {final_val_mse:.6f}")
    print(f"Final Val CosSim: {final_val_cos_sim:.6f}")

    writer.close()
    print("\n✅ Training finished successfully!")
    print(f"Best model saved to: {BEST_MODEL_PATH}")
    print(f"Logs saved to: {LOG_DIR}")

if __name__ == "__main__":
    freeze_support()
    main()