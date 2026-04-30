import os
import csv
import gc
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np


########################################
#             配置模块（可扩展）
########################################

DATASETS = {
    "dsstox": {
        "train_input": r"D:\实验\数据\DSSTox_Feb_2024\train_cleaned.csv",
        "val_input":   r"D:\实验\数据\DSSTox_Feb_2024\val_cleaned.csv",
        "output_dir":  r"D:\实验\数据\DSSTox_Feb_2024\fingerprints"
    },
    # 未来如添加新数据库，只需在此添加路径即可
}

FP_BITS = 1024
CHUNK_SIZE = 2000   # 一次处理2000条，避免内存爆炸


########################################
#      SMILES → 1024-bit Fingerprint
########################################

def smiles_to_fp(smiles: str):
    """
    输入 SMILES，输出 numpy 数组形式的 1024 向量。
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS)
        arr = np.zeros((FP_BITS,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return None


########################################
#     初始化输出文件（写表头）
########################################

def init_output_files(out_dir):
    """
    为 train_fp.csv、val_fp.csv 创建表头。
    """
    os.makedirs(out_dir, exist_ok=True)

    train_fp = os.path.join(out_dir, "train_fp.csv")
    val_fp   = os.path.join(out_dir, "val_fp.csv")

    header = ["canonical_smiles", "formula"] + [f"fp_{i}" for i in range(FP_BITS)]

    for p in [train_fp, val_fp]:
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    return train_fp, val_fp


########################################
#     处理单个数据文件（train 或 val）
########################################

def process_split(split_name, input_path, output_path):
    """
    输入：train_cleaned.csv 或 val_cleaned.csv
    输出：train_fp.csv 或 val_fp.csv
    """
    print(f"\n▶ 正在处理 {split_name} 文件:")
    print(f"读取: {input_path}")

    reader = pd.read_csv(input_path, chunksize=CHUNK_SIZE)
    total_ok = 0
    total_fail = 0

    for chunk in reader:
        chunk = chunk.dropna(subset=["canonical_smiles", "formula"])

        for _, row in tqdm(chunk.iterrows(), total=len(chunk), desc=f"{split_name} chunk processing"):

            smiles = row["canonical_smiles"]
            formula = row["formula"]

            fp = smiles_to_fp(smiles)
            if fp is None:
                total_fail += 1
                continue

            out_row = [smiles, formula] + fp.tolist()

            with open(output_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(out_row)

            total_ok += 1

        del chunk
        gc.collect()

    print(f"✔ {split_name} 完成: 成功={total_ok}, 失败={total_fail}")


########################################
#           主程序入口
########################################

if __name__ == "__main__":
    print("\n====== 开始分子指纹投影任务 ======\n")

    for name, cfg in DATASETS.items():
        print(f"\n====== 数据库: {name} ======\n")

        train_fp, val_fp = init_output_files(cfg["output_dir"])

        # 处理训练集
        process_split("train", cfg["train_input"], train_fp)

        # 处理验证集
        process_split("val", cfg["val_input"], val_fp)

    print("\n====== 全部数据库处理完毕！======\n")
