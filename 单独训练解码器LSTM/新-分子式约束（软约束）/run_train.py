import torch
import pandas as pd

from dataset import MoleculeDataset
from tokenizer import SmilesTokenizer
from LSTMDecoder import LSTMDecoder
from train import train


def load_total_csv(csv_path, emb_dim=1024, ms_dim=22):
    """
    Load merged total CSV and convert to list[dict]
    """
    df = pd.read_csv(csv_path)

    emb_cols = [f"y_{i}" for i in range(emb_dim)]
    ms_cols = [f"ms_{i}" for i in range(ms_dim)]

    required_cols = ["canonical_smiles", "formula"] + emb_cols + ms_cols
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{csv_path} 缺失必要列: {col}")

    data_list = []
    for _, row in df.iterrows():
        data_list.append({
            "smiles": row["canonical_smiles"],
            "formula": row["formula"],
            "mlp_vec": row[emb_cols].values.astype(float).tolist(),
            "ms_vec": row[ms_cols].values.astype(float).tolist(),
        })

    return data_list


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------
    # 1. 加载 train / val 总文件
    # ------------------------------------------------
    train_data = load_total_csv(
        r"D:\实验\数据\DSSTox_Feb_2024\试验小样本5000\lstm_total\train_total.csv"
    )

    val_data = load_total_csv(
        r"D:\实验\数据\DSSTox_Feb_2024\试验小样本5000\lstm_total\val_total.csv"
    )

    # ------------------------------------------------
    # 2. tokenizer
    # ------------------------------------------------
    tokenizer = SmilesTokenizer.load(
        r"C:\Users\Administrator\PyCharmMiscProject\单独训练解码器LSTM\artifacts\tokenizer.json"
    )

    # ------------------------------------------------
    # 3. dataset
    # ------------------------------------------------
    train_dataset = MoleculeDataset(
        data_list=train_data,
        tokenizer=tokenizer,
        cond_dim=1046,
        max_length=128,
    )

    val_dataset = MoleculeDataset(
        data_list=val_data,
        tokenizer=tokenizer,
        cond_dim=1046,
        max_length=128,
    )

    # ------------------------------------------------
    # 4. model
    # ------------------------------------------------
    decoder = LSTMDecoder(
        vocab_size=tokenizer.vocab_size,
        cond_dim=1046,
        num_heavy_atoms=9,
    )

    # ------------------------------------------------
    # 5. train
    # ------------------------------------------------
    train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        decoder=decoder,
        device=device,
        batch_size=128,
        num_epochs=3,
    )


if __name__ == "__main__":
    main()