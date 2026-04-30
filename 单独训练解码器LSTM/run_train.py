# run_train.py
import torch
import pandas as pd

from dataset import MoleculeDataset
from tokenizer import SmilesTokenizer
from projector import FingerprintProjector
from LSTMDecoder import LSTMDecoder
from train import train


def load_fp_csv(csv_path, fp_dim=1024):
    """
    Load fingerprint CSV and convert to list[dict]
    """
    df = pd.read_csv(csv_path)

    fp_cols = [f"fp_{i}" for i in range(fp_dim)]

    data_list = []
    for _, row in df.iterrows():
        data_list.append({
            "smiles": row["canonical_smiles"],
            "formula": row["formula"],
            "fingerprint": row[fp_cols].values.astype(int).tolist()
        })

    return data_list


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------
    # 1. 加载 CSV 数据（真实路径）
    # ------------------------------------------------
    train_data = load_fp_csv(
        r"D:\实验\数据\DSSTox_Feb_2024\fingerprints\train_fp.csv"
    )

    val_data = load_fp_csv(
        r"D:\实验\数据\DSSTox_Feb_2024\fingerprints\val_fp.csv"
    )

    # ------------------------------------------------
    # 2. tokenizer
    # ------------------------------------------------
    tokenizer = SmilesTokenizer.load(
        r"artifacts\tokenizer.json"
    )

    # ------------------------------------------------
    # 3. dataset
    # ------------------------------------------------
    train_dataset = MoleculeDataset(
        data_list=train_data,
        tokenizer=tokenizer,
        max_length=128,
    )

    val_dataset = MoleculeDataset(
        data_list=val_data,
        tokenizer=tokenizer,
        max_length=128,
    )

    # ------------------------------------------------
    # 4. models
    # ------------------------------------------------
    projector = FingerprintProjector(
        fp_dim=1024,
        cond_dim=1024,
    )

    decoder = LSTMDecoder(
        vocab_size=tokenizer.vocab_size,
        cond_dim=1024,
        num_heavy_atoms=9,
    )

    # ------------------------------------------------
    # 5. train
    # ------------------------------------------------
    train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        projector=projector,
        decoder=decoder,
        device=device,
        batch_size=128,
        num_epochs=3,
    )


if __name__ == "__main__":
    main()
