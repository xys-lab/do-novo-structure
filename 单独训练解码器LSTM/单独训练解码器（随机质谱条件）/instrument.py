import pandas as pd
import os

INPUT_FILE = r"D:\实验\数据\DSSTox_Feb_2024\试验小样本5000\top10_adducts_5000.csv"
OUTPUT_FILE = r"D:\实验\数据\DSSTox_Feb_2024\试验小样本5000\instrument_10.csv"
CHUNK_SIZE = 1_000_000


def expand_instrument(input_file, output_file, chunk_size=1_000_000):
    # 分块读取 CSV
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        # 为每行生成 Orbitrap 和 QTOF 两行
        chunk_orbitrap = chunk.copy()
        chunk_orbitrap["INSTRUMENT"] = "Orbitrap"

        chunk_qtof = chunk.copy()
        chunk_qtof["INSTRUMENT"] = "QTOF"

        # 拼接两种 instrument
        chunk_expanded = pd.concat([chunk_orbitrap, chunk_qtof], ignore_index=True)

        # 写入 CSV
        if i == 0:
            chunk_expanded.to_csv(output_file, index=False)
        else:
            chunk_expanded.to_csv(output_file, index=False, header=False, mode="a")

        print(f"Processed chunk {i+1}")

    return output_file


if __name__ == "__main__":
    expand_instrument(INPUT_FILE, OUTPUT_FILE, CHUNK_SIZE)