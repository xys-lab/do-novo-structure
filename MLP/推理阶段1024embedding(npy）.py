import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from numpy.lib.format import open_memmap

# =====================================================
# 配置区
# =====================================================
MODEL_PATH = r"D:\实验\数据\Nist数据\MLP\1008403\branch_mlp_group_split_metrics\best_model.pt"

FP_FILE = r"D:\实验\数据\Nist数据\MLP\1008403\fp.npy"
MS_FILE = r"D:\实验\数据\Nist数据\MLP\1008403\ms.npy"

# 真实 ID 文件（推荐是 .npy）
# 例如内容可以是：
# [10001, 10002, 10003, ...]
# 或 ['ID_1', 'ID_2', ...]
#
# 如果没有真实 ID 文件，就设为 None
REAL_ID_FILE = r"D:\实验\数据\Nist数据\MLP\1008403\id.npy"
# REAL_ID_FILE = None

OUTPUT_DIR = r"D:\实验\数据\Nist数据\MLP\1008403\inference_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 输出文件
EMBEDDING_FILE = os.path.join(OUTPUT_DIR, "pred_embedding.npy")
REAL_ID_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pred_real_id.npy")
INDEX_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pred_index.npy")

# 小规模测试时可设成 100 / 1000 / 5000
# 全量跑时设为 None
MAX_SAMPLES = 5000

# 根据显存调节
BATCH_SIZE = 2048

# Windows 下一般先用 0 最稳；后续可尝试 2/4
NUM_WORKERS = 0

# True 时启用半精度自动混合推理（GPU 上更快更省显存）
USE_AMP = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# 模型结构（必须与训练时完全一致）
# =====================================================
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


# =====================================================
# Dataset
# =====================================================
class InferDataset(Dataset):
    def __init__(self, fp_array, ms_array, id_array, n_samples):
        self.fp = fp_array
        self.ms = ms_array
        self.ids = id_array   # 可以为 None
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 不做全量拷贝，只按索引取单条
        fp = np.asarray(self.fp[idx], dtype=np.float32)
        ms = np.asarray(self.ms[idx], dtype=np.float32)

        # 保存行号索引，保证永远可追踪
        index_value = np.int64(idx)

        # 如果有真实 ID，则一并返回
        if self.ids is None:
            real_id = ""
        else:
            real_id = self.ids[idx]

        return index_value, real_id, fp, ms


# =====================================================
# 工具函数
# =====================================================
def load_real_id_array(real_id_file, n_total):
    """
    读取真实 ID。
    支持：
    1. .npy
    2. .txt（单列）
    3. .csv（单列，简单情况）
    """
    if real_id_file is None:
        return None

    if not os.path.exists(real_id_file):
        raise FileNotFoundError(f"真实 ID 文件不存在: {real_id_file}")

    ext = os.path.splitext(real_id_file)[1].lower()

    if ext == ".npy":
        ids = np.load(real_id_file, mmap_mode="r", allow_pickle=True)
    elif ext == ".txt":
        ids = np.loadtxt(real_id_file, dtype=str, encoding="utf-8")
    elif ext == ".csv":
        # 简单单列 CSV 场景
        ids = np.genfromtxt(real_id_file, delimiter=",", dtype=str, encoding="utf-8")
        if ids.ndim > 1:
            # 如果是二维，默认取第一列
            ids = ids[:, 0]
    else:
        raise ValueError(f"暂不支持的 ID 文件格式: {ext}")

    if len(ids) != n_total:
        raise ValueError(
            f"真实 ID 数量与样本数不一致: len(ids)={len(ids)}, n_total={n_total}"
        )

    return ids


def build_real_id_memmap(path, n_use, sample_ids):
    """
    根据真实 ID 的类型创建输出文件。
    如果是数字类型，尽量保留数字；
    如果是字符串，使用 unicode 字符串数组保存。
    """
    if sample_ids is None:
        return None, None

    sample0 = sample_ids[0]

    # numpy 标量 / python 标量都统一判断
    if isinstance(sample0, (np.integer, int)):
        arr = open_memmap(path, mode="w+", dtype=np.int64, shape=(n_use,))
        return arr, "int64"

    if isinstance(sample0, (np.floating, float)):
        arr = open_memmap(path, mode="w+", dtype=np.float64, shape=(n_use,))
        return arr, "float64"

    # 其余一律按字符串存
    max_len = max(len(str(x)) for x in sample_ids[: min(1000, len(sample_ids))])
    max_len = max(max_len, 16)
    dtype = f"<U{max_len}"

    arr = open_memmap(path, mode="w+", dtype=dtype, shape=(n_use,))
    return arr, dtype


# =====================================================
# 主程序
# =====================================================
def main():
    print("=" * 60)
    print("Step 1/6: Loading input arrays with mmap")
    print("=" * 60)

    fp_all = np.load(FP_FILE, mmap_mode="r")
    ms_all = np.load(MS_FILE, mmap_mode="r")

    print("Original fp shape:", fp_all.shape, "dtype:", fp_all.dtype)
    print("Original ms shape:", ms_all.shape, "dtype:", ms_all.dtype)

    assert fp_all.shape[0] == ms_all.shape[0], "fp 和 ms 样本数不一致"
    assert fp_all.shape[1] == 1002, f"fp 维度不是1002，而是 {fp_all.shape[1]}"
    assert ms_all.shape[1] == 22, f"ms 维度不是22，而是 {ms_all.shape[1]}"

    n_total = fp_all.shape[0]

    print("=" * 60)
    print("Step 2/6: Loading real IDs")
    print("=" * 60)

    real_ids = load_real_id_array(REAL_ID_FILE, n_total)

    if real_ids is None:
        print("No real ID file provided. Will save row index only.")
    else:
        print("Real ID loaded successfully.")
        print("Real ID type:", type(real_ids[0]).__name__)

    if MAX_SAMPLES is None:
        n_use = n_total
        print(f"Full mode: use all {n_use} samples")
    else:
        n_use = min(MAX_SAMPLES, n_total)
        print(f"Testing mode: use first {n_use} samples")

    print("=" * 60)
    print("Step 3/6: Building DataLoader")
    print("=" * 60)

    dataset = InferDataset(fp_all, ms_all, real_ids, n_use)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    print("Device:", DEVICE)
    print("Batch size:", BATCH_SIZE)
    print("Num workers:", NUM_WORKERS)
    print("Total batches:", len(loader))
    print("AMP enabled:", USE_AMP and DEVICE.type == "cuda")

    print("=" * 60)
    print("Step 4/6: Loading trained model")
    print("=" * 60)

    model = BranchMLP().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully.")

    print("=" * 60)
    print("Step 5/6: Preparing output memmap files")
    print("=" * 60)

    # 1024维 embedding 输出
    emb_out = open_memmap(
        EMBEDDING_FILE,
        mode="w+",
        dtype=np.float32,
        shape=(n_use, 1024)
    )

    # 行号索引输出（永远建议保存）
    index_out = open_memmap(
        INDEX_OUTPUT_FILE,
        mode="w+",
        dtype=np.int64,
        shape=(n_use,)
    )

    # 真实 ID 输出（如果有）
    if real_ids is not None:
        id_out, id_dtype = build_real_id_memmap(
            REAL_ID_OUTPUT_FILE,
            n_use,
            real_ids[:n_use]
        )
        print("Real ID output file:", REAL_ID_OUTPUT_FILE)
        print("Real ID output dtype:", id_dtype)
    else:
        id_out = None
        print("Real ID output skipped.")

    print("Embedding output file:", EMBEDDING_FILE)
    print("Index output file:", INDEX_OUTPUT_FILE)

    print("=" * 60)
    print("Step 6/6: Running inference and writing to disk batch-by-batch")
    print("=" * 60)

    written = 0

    with torch.no_grad():
        for batch_index, batch_real_id, fp_batch, ms_batch in tqdm(loader, desc="Inference"):
            fp_batch = fp_batch.to(DEVICE, non_blocking=True)
            ms_batch = ms_batch.to(DEVICE, non_blocking=True)

            if USE_AMP and DEVICE.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    emb_batch = model(fp_batch, ms_batch)
            else:
                emb_batch = model(fp_batch, ms_batch)

            emb_batch = emb_batch.float().cpu().numpy().astype(np.float32)
            batch_index_np = batch_index.numpy().astype(np.int64)

            bsz = emb_batch.shape[0]
            start = written
            end = written + bsz

            # 边推理边写盘
            emb_out[start:end] = emb_batch
            index_out[start:end] = batch_index_np

            if id_out is not None:
                # DataLoader 对字符串 batch 会整理成 list/tuple
                if isinstance(batch_real_id, (list, tuple)):
                    id_out[start:end] = np.asarray(batch_real_id, dtype=id_out.dtype)
                else:
                    id_out[start:end] = batch_real_id.numpy().astype(id_out.dtype)

            written = end

            # 定期 flush，防止中断损失过多
            if written % (BATCH_SIZE * 20) == 0 or written == n_use:
                emb_out.flush()
                index_out.flush()
                if id_out is not None:
                    id_out.flush()

    emb_out.flush()
    index_out.flush()
    if id_out is not None:
        id_out.flush()

    print("Inference finished.")
    print("Total written samples:", written)
    print("Embedding shape saved:", (n_use, 1024))
    print("Done.")


if __name__ == "__main__":
    main()