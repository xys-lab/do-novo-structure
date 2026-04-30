# prepare_lstm_input.py
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from tokenizer import SmilesTokenizer


# =============================
# 配置文件路径
# =============================
mlp_csv_path = Path(r"F:\从头生成任务1\训练\file1_300000.csv")   # ID + y_0~y_1023
ms_csv_path = Path(r"F:\从头生成任务1\训练\file2_300000.csv")    # ID + ms_0~ms_21
smiles_csv_path = Path(r"F:\从头生成任务1\训练\file3_300000.csv")  # ID + canonical_smiles + exact_mass + inchikey

tokenizer_path = Path(
    r"C:\Users\Administrator\PyCharmMiscProject\单独训练解码器LSTM\artifacts\tokenizer.json"
)

output_dir = Path(r"F:\从头生成任务1\训练\lstm_memmap")
train_dir = output_dir / "train"
val_dir = output_dir / "val"

# =============================
# 配置
# =============================
TRAIN_RATIO = 0.9
RANDOM_SEED = 42
MAX_LENGTH = 128
EMB_DIM = 1024
MS_DIM = 22
COND_DIM = EMB_DIM + MS_DIM


def write_split_memmap(
    smiles_arr,
    exact_mass_arr,
    mlp_mat,
    ms_mat,
    out_dir: Path,
    tokenizer: SmilesTokenizer,
    max_length: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(smiles_arr)
    seq_len = max_length - 1

    if not (len(exact_mass_arr) == len(mlp_mat) == len(ms_mat) == n):
        raise ValueError("split 内部长度不一致，请检查。")

    input_ids_mm = open_memmap(
        out_dir / "input_ids.npy",
        mode="w+",
        dtype=np.int32,
        shape=(n, seq_len),
    )
    target_ids_mm = open_memmap(
        out_dir / "target_ids.npy",
        mode="w+",
        dtype=np.int32,
        shape=(n, seq_len),
    )
    token_mask_mm = open_memmap(
        out_dir / "token_mask.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(n, seq_len),
    )
    cond_mm = open_memmap(
        out_dir / "cond.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n, COND_DIM),
    )
    exact_mass_mm = open_memmap(
        out_dir / "exact_mass.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n,),
    )

    for i in range(n):
        smiles = str(smiles_arr[i])
        ids = tokenizer.encode_padded(smiles, max_length=max_length)

        inp = np.asarray(ids[:-1], dtype=np.int32)
        tgt = np.asarray(ids[1:], dtype=np.int32)
        mask = (tgt != tokenizer.pad_token_id).astype(np.uint8)

        input_ids_mm[i] = inp
        target_ids_mm[i] = tgt
        token_mask_mm[i] = mask

        cond_mm[i, :EMB_DIM] = mlp_mat[i]
        cond_mm[i, EMB_DIM:] = ms_mat[i]
        exact_mass_mm[i] = exact_mass_arr[i]

        if (i + 1) % 50000 == 0:
            print(f"[INFO] {out_dir.name}: 已写入 {i + 1}/{n}")

    input_ids_mm.flush()
    target_ids_mm.flush()
    token_mask_mm.flush()
    cond_mm.flush()
    exact_mass_mm.flush()

    meta = {
        "num_samples": int(n),
        "max_length": int(max_length),
        "seq_len": int(seq_len),
        "emb_dim": int(EMB_DIM),
        "ms_dim": int(MS_DIM),
        "cond_dim": int(COND_DIM),
        "input_ids_dtype": "int32",
        "target_ids_dtype": "int32",
        "token_mask_dtype": "uint8",
        "cond_dtype": "float32",
        "exact_mass_dtype": "float32",
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    random.seed(RANDOM_SEED)

    print("\n[INFO] 正在读取输入文件...")
    mlp_df = pd.read_csv(mlp_csv_path, low_memory=False)
    ms_df = pd.read_csv(ms_csv_path, low_memory=False)
    smiles_df = pd.read_csv(smiles_csv_path, low_memory=False)

    print(f"[INFO] mlp_df 行数:    {len(mlp_df)}")
    print(f"[INFO] ms_df 行数:     {len(ms_df)}")
    print(f"[INFO] smiles_df 行数: {len(smiles_df)}")

    # =============================
    # 基础检查
    # =============================
    required_smiles_cols = ["ID", "canonical_smiles", "exact_mass", "inchikey"]
    for col in required_smiles_cols:
        if col not in smiles_df.columns:
            raise ValueError(f"smiles 文件缺失必要列: {col}")

    emb_cols = [f"y_{i}" for i in range(EMB_DIM)]
    ms_cols = [f"ms_{i}" for i in range(MS_DIM)]

    for col in emb_cols:
        if col not in mlp_df.columns:
            raise ValueError(f"MLP 文件缺失列: {col}")

    for col in ms_cols:
        if col not in ms_df.columns:
            raise ValueError(f"MS 文件缺失列: {col}")

    # =============================
    # ID 对齐检查
    # =============================
    print("[INFO] 正在检查三个文件的 ID 是否严格一致...")
    if len(mlp_df) != len(ms_df) or len(mlp_df) != len(smiles_df):
        raise ValueError("三个文件行数不一致，请检查。")

    if not (mlp_df["ID"].equals(ms_df["ID"]) and mlp_df["ID"].equals(smiles_df["ID"])):
        raise ValueError("三个文件的 ID 列不完全一致或顺序不一致，请检查！")

    print("[OK] 三个文件的 ID 已严格对齐。")

    # =============================
    # tokenizer
    # =============================
    print("[INFO] 正在加载 tokenizer...")
    tokenizer = SmilesTokenizer.load(str(tokenizer_path))

    # =============================
    # 类型清洗
    # =============================
    print("[INFO] 正在进行字段清洗...")

    smiles_base = smiles_df[["ID", "canonical_smiles", "exact_mass", "inchikey"]].copy()
    smiles_base["exact_mass"] = pd.to_numeric(smiles_base["exact_mass"], errors="coerce")

    mlp_numeric = mlp_df[emb_cols].apply(pd.to_numeric, errors="coerce")
    ms_numeric = ms_df[ms_cols].apply(pd.to_numeric, errors="coerce")

    valid_mask = (
        smiles_base["canonical_smiles"].notna()
        & smiles_base["exact_mass"].notna()
        & mlp_numeric.notna().all(axis=1)
        & ms_numeric.notna().all(axis=1)
    )

    removed = int((~valid_mask).sum())
    if removed > 0:
        print(f"[WARNING] 有 {removed} 条样本因缺失/非法数值被移除。")

    smiles_base = smiles_base.loc[valid_mask].reset_index(drop=True)
    smiles_base["canonical_smiles"] = smiles_base["canonical_smiles"].astype(str)

    mlp_numeric = mlp_numeric.loc[valid_mask].reset_index(drop=True)
    ms_numeric = ms_numeric.loc[valid_mask].reset_index(drop=True)

    # 转 numpy
    smiles_arr = smiles_base["canonical_smiles"].to_numpy()
    exact_mass_arr = smiles_base["exact_mass"].to_numpy(dtype=np.float32)
    inchikey_series = smiles_base["inchikey"]

    mlp_mat = mlp_numeric.to_numpy(dtype=np.float32, copy=False)
    ms_mat = ms_numeric.to_numpy(dtype=np.float32, copy=False)

    # =============================
    # 按 inchikey 划分 train / val
    # =============================
    print("[INFO] 正在按 inchikey 进行 train/val 划分...")

    unique_inchikeys = inchikey_series.dropna().astype(str).unique().tolist()
    random.shuffle(unique_inchikeys)

    split_idx = int(len(unique_inchikeys) * TRAIN_RATIO)
    train_keys = set(unique_inchikeys[:split_idx])
    val_keys = set(unique_inchikeys[split_idx:])

    train_mask = inchikey_series.astype(str).isin(train_keys)
    val_mask = inchikey_series.astype(str).isin(val_keys)

    missing_inchikey_mask = inchikey_series.isna()
    if int(missing_inchikey_mask.sum()) > 0:
        print(f"[WARNING] 有 {int(missing_inchikey_mask.sum())} 条样本缺失 inchikey，默认放入训练集。")
        train_mask = train_mask | missing_inchikey_mask

    train_smiles = smiles_arr[train_mask.to_numpy()]
    train_exact_mass = exact_mass_arr[train_mask.to_numpy()]
    train_mlp = mlp_mat[train_mask.to_numpy()]
    train_ms = ms_mat[train_mask.to_numpy()]

    val_smiles = smiles_arr[val_mask.to_numpy()]
    val_exact_mass = exact_mass_arr[val_mask.to_numpy()]
    val_mlp = mlp_mat[val_mask.to_numpy()]
    val_ms = ms_mat[val_mask.to_numpy()]

    print(f"[INFO] 训练样本数: {len(train_smiles)}")
    print(f"[INFO] 验证样本数: {len(val_smiles)}")

    # =============================
    # 写入 memmap
    # =============================
    print("[INFO] 正在写入 train memmap...")
    write_split_memmap(
        smiles_arr=train_smiles,
        exact_mass_arr=train_exact_mass,
        mlp_mat=train_mlp,
        ms_mat=train_ms,
        out_dir=train_dir,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    print("[INFO] 正在写入 val memmap...")
    write_split_memmap(
        smiles_arr=val_smiles,
        exact_mass_arr=val_exact_mass,
        mlp_mat=val_mlp,
        ms_mat=val_ms,
        out_dir=val_dir,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_valid_samples": int(len(smiles_base)),
        "train_samples": int(len(train_smiles)),
        "val_samples": int(len(val_smiles)),
        "train_ratio": float(len(train_smiles) / len(smiles_base)) if len(smiles_base) > 0 else 0.0,
        "val_ratio": float(len(val_smiles) / len(smiles_base)) if len(smiles_base) > 0 else 0.0,
        "unique_inchikey_total": int(len(unique_inchikeys)),
        "train_inchikey_count": int(len(train_keys)),
        "val_inchikey_count": int(len(val_keys)),
        "max_length": int(MAX_LENGTH),
        "cond_dim": int(COND_DIM),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[OK] 预处理完成，已生成 memmap 数据。")
    print(f"     train -> {train_dir}")
    print(f"     val   -> {val_dir}")
    print(f"     汇总  -> {output_dir / 'summary.json'}\n")


if __name__ == "__main__":
    main()