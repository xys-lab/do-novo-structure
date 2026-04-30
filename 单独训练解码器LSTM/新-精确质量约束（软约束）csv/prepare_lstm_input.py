import pandas as pd
import random
from pathlib import Path

# =============================
# 配置文件路径
# =============================
mlp_csv_path = Path(r"F:\从头生成任务1\训练\file1_300000.csv")
ms_csv_path = Path(r"F:\从头生成任务1\训练\file2_300000.csv")
canonical_smiles_csv_path = Path(r"F:\从头生成任务1\训练\file3_300000.csv")

output_dir = Path(r"F:\从头生成任务1\训练\lstm_total")
output_dir.mkdir(parents=True, exist_ok=True)

all_output_csv_path = output_dir / "all_total.csv"
train_output_csv_path = output_dir / "train_total.csv"
val_output_csv_path = output_dir / "val_total.csv"

# =============================
# 配置
# =============================
TRAIN_RATIO = 0.9
RANDOM_SEED = 42

# =============================
# 读取 CSV
# =============================
print("\n[INFO] 正在读取输入文件...")

mlp_df = pd.read_csv(mlp_csv_path)                 # ID + y_0~y_1023
ms_df = pd.read_csv(ms_csv_path)                   # ID + ms_0~ms_21
smiles_df = pd.read_csv(canonical_smiles_csv_path) # ID + canonical_smiles + formula + inchikey

print(f"[INFO] mlp_df 行数: {len(mlp_df)}")
print(f"[INFO] ms_df 行数: {len(ms_df)}")
print(f"[INFO] smiles_df 行数: {len(smiles_df)}")

# =============================
# 基础检查
# =============================
required_smiles_cols = ["ID", "canonical_smiles", "exact_mass", "inchikey"]
for col in required_smiles_cols:
    if col not in smiles_df.columns:
        raise ValueError(f"smiles 文件缺失必要列: {col}")

cond_cols = [f"y_{i}" for i in range(1024)]
ms_cols = [f"ms_{i}" for i in range(22)]

for col in cond_cols:
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
# 合并总表
# =============================
print("[INFO] 正在合并总表...")

all_df = pd.concat(
    [
        smiles_df[["ID", "canonical_smiles", "exact_mass", "inchikey"]].reset_index(drop=True),
        mlp_df[cond_cols].reset_index(drop=True),
        ms_df[ms_cols].reset_index(drop=True),
    ],
    axis=1
)

all_df.to_csv(all_output_csv_path, index=False, encoding="utf-8-sig")
print(f"[OK] 已保存总表: {all_output_csv_path}")

# =============================
# 按 inchikey 分组划分 train / val
# =============================
print("[INFO] 正在按 inchikey 进行 train/val 划分...")

unique_inchikeys = all_df["inchikey"].dropna().astype(str).unique().tolist()
random.seed(RANDOM_SEED)
random.shuffle(unique_inchikeys)

split_idx = int(len(unique_inchikeys) * TRAIN_RATIO)
train_keys = set(unique_inchikeys[:split_idx])
val_keys = set(unique_inchikeys[split_idx:])

train_df = all_df[all_df["inchikey"].astype(str).isin(train_keys)].copy()
val_df = all_df[all_df["inchikey"].astype(str).isin(val_keys)].copy()

# 对缺失 inchikey 的样本做保护
missing_inchikey_df = all_df[all_df["inchikey"].isna()].copy()
if len(missing_inchikey_df) > 0:
    print(f"[WARNING] 有 {len(missing_inchikey_df)} 条样本缺失 inchikey，默认放入训练集。")
    train_df = pd.concat([train_df, missing_inchikey_df], axis=0)

# 重置索引
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# =============================
# 保存 train / val
# =============================
train_df.to_csv(train_output_csv_path, index=False, encoding="utf-8-sig")
val_df.to_csv(val_output_csv_path, index=False, encoding="utf-8-sig")

print(f"[OK] train 文件已保存: {train_output_csv_path}")
print(f"[OK] val 文件已保存:   {val_output_csv_path}")

print("\n========== 划分统计 ==========")
print(f"总样本数:   {len(all_df)}")
print(f"训练样本数: {len(train_df)}")
print(f"验证样本数: {len(val_df)}")
print(f"训练比例:   {len(train_df) / len(all_df):.4f}")
print(f"验证比例:   {len(val_df) / len(all_df):.4f}")
print(f"唯一 inchikey 总数: {len(unique_inchikeys)}")
print(f"train inchikey 数:  {len(train_keys)}")
print(f"val inchikey 数:    {len(val_keys)}")
print("==============================\n")