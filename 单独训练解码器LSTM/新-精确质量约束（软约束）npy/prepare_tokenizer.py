# prepare_tokenizer.py
from pathlib import Path
from typing import List, Dict

import pandas as pd

from tokenizer import SmilesTokenizer


########################################
#        数据集配置（你只改这里）
########################################

DATASETS: Dict[str, Dict] = {
    "dsstox": {
        "files": [
            Path(r"D:\实验\数据\DSSTox_Feb_2024\fingerprints\train_fp.csv"),
            Path(r"D:\实验\数据\DSSTox_Feb_2024\fingerprints\val_fp.csv"),
        ],
        "smiles_col": "canonical_smiles",
    },

    # 以后新增数据集可以继续往下加
    # "pubchem": {
    #     "files": [
    #         Path(r"D:\data\pubchem\pubchem_cleaned.csv"),
    #     ],
    #     "smiles_col": "canonical_smiles",
    # },
}


def load_smiles_from_csv(
    file_path: Path,
    smiles_col: str,
    chunksize: int = 200000
) -> List[str]:
    """
    分块读取，避免一次性把整个文件都读进内存
    """
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    header = pd.read_csv(file_path, nrows=0)
    if smiles_col not in header.columns:
        raise ValueError(f"{file_path.name} 中缺失 SMILES 列: {smiles_col}")

    smiles: List[str] = []
    for chunk in pd.read_csv(
        file_path,
        usecols=[smiles_col],
        chunksize=chunksize,
        low_memory=False,
    ):
        smiles.extend(
            chunk[smiles_col].dropna().astype(str).tolist()
        )

    return smiles


def main():
    all_smiles: List[str] = []

    print("\n========== 构建 SMILES Tokenizer ==========\n")

    for dataset_name, cfg in DATASETS.items():
        print(f"[INFO] 加载数据集: {dataset_name}")

        smiles_col = cfg["smiles_col"]

        for file_path in cfg["files"]:
            print(f"  -> 文件: {file_path}")
            smiles = load_smiles_from_csv(file_path, smiles_col)
            print(f"     读取 SMILES 数量: {len(smiles)}")
            all_smiles.extend(smiles)

    print("\n------------------------------------------")
    print(f"[INFO] SMILES 总数量: {len(all_smiles)}")
    print("------------------------------------------\n")

    tokenizer = SmilesTokenizer()
    tokenizer.build_vocab(all_smiles, min_freq=1)

    output_path = Path("artifacts/tokenizer.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_path)

    print("[OK] Tokenizer 构建完成")
    print(f"     词表大小: {tokenizer.vocab_size}")
    print(f"     保存路径: {output_path.resolve()}\n")


if __name__ == "__main__":
    main()