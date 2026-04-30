import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# =====================================================
# 配置区
# =====================================================
MODEL_PATH = r"D:\实验\数据\Nist数据\MLP\1008403训练MLP模型\branch_mlp_group_split_metrics\best_model.pt"

FP_CSV = r"F:\从头生成任务1\训练\morgan_fp_1002_000001.csv"
MS_CSV = r"F:\从头生成任务1\训练\all_features_000001.csv"

OUTPUT_CSV = r"F:\从头生成任务1\训练\pred_embedding_000001.csv"

# 前期测试时可设小一点，比如 1000 / 5000
# 全量跑时设为 None
MAX_SAMPLES = None

# 每次从 CSV 读取多少行
CHUNK_SIZE = 20000

# 每个 batch 送入模型多少样本
BATCH_SIZE = 2048

# GPU 自动混合精度
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
# 工具函数
# =====================================================
def build_column_names():
    fp_cols = ["ID"] + [f"fp_{i}" for i in range(1002)]
    ms_cols = ["ID"] + [f"ms_{i}" for i in range(22)]
    out_cols = ["ID"] + [f"y_{i}" for i in range(1024)]
    return fp_cols, ms_cols, out_cols


def validate_columns(df, expected_cols, file_name):
    actual_cols = list(df.columns)
    if actual_cols != expected_cols:
        raise ValueError(
            f"{file_name} 列名不符合预期。\n"
            f"期望前几列: {expected_cols[:5]} ... 最后几列: {expected_cols[-5:]}\n"
            f"实际前几列: {actual_cols[:5]} ... 最后几列: {actual_cols[-5:]}"
        )


def run_model_in_batches(model, fp_np, ms_np, batch_size, device, use_amp):
    outputs = []
    n = fp_np.shape[0]

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            fp_batch = torch.from_numpy(fp_np[start:end]).to(device, non_blocking=True)
            ms_batch = torch.from_numpy(ms_np[start:end]).to(device, non_blocking=True)

            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(fp_batch, ms_batch)
            else:
                pred = model(fp_batch, ms_batch)

            outputs.append(pred.float().cpu().numpy().astype(np.float32))

    return np.concatenate(outputs, axis=0)


# =====================================================
# 主函数
# =====================================================
def main():
    print("=" * 60)
    print("Step 1/6: Preparing model")
    print("=" * 60)

    model = BranchMLP().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully.")
    print("Device:", DEVICE)
    print("AMP enabled:", USE_AMP and DEVICE.type == "cuda")

    print("=" * 60)
    print("Step 2/6: Preparing column names")
    print("=" * 60)

    fp_cols, ms_cols, out_cols = build_column_names()

    # 先删除旧文件，避免追加混乱
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
        print("Old output file removed:", OUTPUT_CSV)

    print("=" * 60)
    print("Step 3/6: Building CSV chunk readers")
    print("=" * 60)

    fp_reader = pd.read_csv(FP_CSV, chunksize=CHUNK_SIZE)
    ms_reader = pd.read_csv(MS_CSV, chunksize=CHUNK_SIZE)

    total_written = 0
    chunk_idx = 0
    header_written = False

    print("=" * 60)
    print("Step 4/6: Start chunk-by-chunk inference")
    print("=" * 60)

    for fp_chunk, ms_chunk in tqdm(zip(fp_reader, ms_reader), desc="Chunks"):
        chunk_idx += 1

        # 列名校验
        validate_columns(fp_chunk, fp_cols, "FP_CSV")
        validate_columns(ms_chunk, ms_cols, "MS_CSV")

        # 行数校验
        if len(fp_chunk) != len(ms_chunk):
            raise ValueError(
                f"第 {chunk_idx} 个分块行数不一致: fp_chunk={len(fp_chunk)}, ms_chunk={len(ms_chunk)}"
            )

        # 如果设置了 MAX_SAMPLES，要截断最后一个块
        if MAX_SAMPLES is not None:
            remain = MAX_SAMPLES - total_written
            if remain <= 0:
                break
            if len(fp_chunk) > remain:
                fp_chunk = fp_chunk.iloc[:remain].copy()
                ms_chunk = ms_chunk.iloc[:remain].copy()

        # 严格校验 ID 一一对应
        fp_ids = fp_chunk["ID"].astype(str).values
        ms_ids = ms_chunk["ID"].astype(str).values

        if not np.array_equal(fp_ids, ms_ids):
            mismatch_idx = np.where(fp_ids != ms_ids)[0][0]
            raise ValueError(
                f"第 {chunk_idx} 个分块 ID 不一致。\n"
                f"首次不一致位置: chunk内第 {mismatch_idx} 行\n"
                f"fp ID = {fp_ids[mismatch_idx]}\n"
                f"ms ID = {ms_ids[mismatch_idx]}"
            )

        # 提取特征矩阵
        fp_np = fp_chunk[[f"fp_{i}" for i in range(1002)]].to_numpy(dtype=np.float32, copy=True)
        ms_np = ms_chunk[[f"ms_{i}" for i in range(22)]].to_numpy(dtype=np.float32, copy=True)

        # shape 校验
        if fp_np.shape[1] != 1002:
            raise ValueError(f"fp 特征维度错误: {fp_np.shape}")
        if ms_np.shape[1] != 22:
            raise ValueError(f"ms 特征维度错误: {ms_np.shape}")

        print(f"\n[Chunk {chunk_idx}] rows = {len(fp_chunk)}")
        print("FP chunk shape:", fp_np.shape)
        print("MS chunk shape:", ms_np.shape)
        print("ID sample:", fp_ids[:3])

        # 模型推理
        pred_np = run_model_in_batches(
            model=model,
            fp_np=fp_np,
            ms_np=ms_np,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            use_amp=USE_AMP
        )

        if pred_np.shape[1] != 1024:
            raise ValueError(f"输出 embedding 维度不是 1024，而是 {pred_np.shape}")

        print("Pred embedding shape:", pred_np.shape)

        # 组装输出 DataFrame
        out_df = pd.DataFrame(pred_np, columns=[f"y_{i}" for i in range(1024)])
        out_df.insert(0, "ID", fp_ids)

        # 写出 CSV
        out_df.to_csv(
            OUTPUT_CSV,
            mode="a",
            header=not header_written,
            index=False,
            encoding="utf-8-sig"
        )

        header_written = True
        total_written += len(out_df)

        print(f"Written rows so far: {total_written}")
        print(f"Output file: {OUTPUT_CSV}")

        if MAX_SAMPLES is not None and total_written >= MAX_SAMPLES:
            break

    print("=" * 60)
    print("Step 5/6: Final check")
    print("=" * 60)
    print("Total written rows:", total_written)
    print("Output saved to:", OUTPUT_CSV)

    print("=" * 60)
    print("Step 6/6: Done")
    print("=" * 60)


if __name__ == "__main__":
    main()