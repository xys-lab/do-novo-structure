# run_train.py
import torch

from dataset import MemmapDataset
from tokenizer import SmilesTokenizer
from LSTMDecoder import LSTMDecoder
from train import train


# ==========================================
# 配置（你主要改这里）
# ==========================================
TOKENIZER_PATH = r"C:\Users\Administrator\PyCharmMiscProject\单独训练解码器LSTM\artifacts\tokenizer.json"

TRAIN_MEMMAP_DIR = r"F:\从头生成任务1\训练\lstm_memmap\train"
VAL_MEMMAP_DIR = r"F:\从头生成任务1\训练\lstm_memmap\val"

SAVE_DIR = r"F:\从头生成任务1\训练\checkpoints_lstm"

BATCH_SIZE = 128
NUM_EPOCHS = 20
LR = 1e-4
LAMBDA_MASS = 0.1

NUM_WORKERS = 0   # Windows 建议先用 0，稳定后可尝试 2/4
PIN_MEMORY = True


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] 正在加载 tokenizer...")
    tokenizer = SmilesTokenizer.load(TOKENIZER_PATH)

    print("[INFO] 正在加载 train / val memmap 数据集...")
    train_dataset = MemmapDataset(TRAIN_MEMMAP_DIR)
    val_dataset = MemmapDataset(VAL_MEMMAP_DIR)

    print("[INFO] 正在初始化模型...")
    decoder = LSTMDecoder(
        vocab_size=tokenizer.vocab_size,
        cond_dim=1046,
        pad_token_id=tokenizer.pad_token_id,
    )

    print("[INFO] 开始训练...\n")
    train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        decoder=decoder,
        device=device,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        lambda_mass=LAMBDA_MASS,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        save_dir=SAVE_DIR,
        use_amp=True,
    )


if __name__ == "__main__":
    main()