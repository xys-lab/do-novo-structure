import pandas as pd
import numpy as np
import os
import re

PROVIDED_FILE = r"D:\实验\数据\DSSTox_Feb_2024\试验小样本5000\instrument_10_5000.csv"
NIST_FILE = r"D:\实验\数据\Nist数据\MLP\nist20数据集（1008403clean）.csv"
OUTPUT_FILE = r"D:\实验\数据\DSSTox_Feb_2024\试验小样本5000\provided_with_CE_5000.csv"

CHUNK_SIZE = 200000
N_SAMPLES = 10


def has_number(val):
    if pd.isna(val):
        return False
    return re.search(r"\d+\.?\d*", str(val)) is not None


def is_orbitrap(val):
    if pd.isna(val):
        return False
    return "NCE" in str(val).upper()


def sample_collision_energy(provided_file, nist_file, output_file, chunk_size=200000, n_samples=10, random_seed=42):
    print("Reading NIST collision energy data...")

    rng = np.random.default_rng(random_seed)

    nist_df = pd.read_csv(
        nist_file,
        usecols=["COLLISION ENERGY"],
        dtype=str,
        keep_default_na=True
    )

    # 分类
    orbitrap_mask = nist_df["COLLISION ENERGY"].apply(is_orbitrap)
    qtof_mask = (~orbitrap_mask) & nist_df["COLLISION ENERGY"].apply(has_number)

    nist_orbitrap = nist_df[orbitrap_mask]
    nist_qtof = nist_df[qtof_mask]

    print("Calculating probability distributions...")

    orbitrap_counts = nist_orbitrap["COLLISION ENERGY"].value_counts()
    orbitrap_values = orbitrap_counts.index.to_numpy()
    orbitrap_probs = orbitrap_counts.values / orbitrap_counts.values.sum()

    qtof_counts = nist_qtof["COLLISION ENERGY"].value_counts()
    qtof_values = qtof_counts.index.to_numpy()
    qtof_probs = qtof_counts.values / qtof_counts.values.sum()

    print("Orbitrap types:", len(orbitrap_values))
    print("QTOF types:", len(qtof_values))
    print("Start processing provided file...")

    for i, chunk in enumerate(pd.read_csv(provided_file, chunksize=chunk_size)):
        expanded_rows = []

        for _, row in chunk.iterrows():
            instrument = str(row["INSTRUMENT"]).strip().lower()

            if instrument == "orbitrap":
                sampled = rng.choice(
                    orbitrap_values,
                    size=n_samples,
                    replace=True,
                    p=orbitrap_probs
                )
            elif instrument == "qtof":
                sampled = rng.choice(
                    qtof_values,
                    size=n_samples,
                    replace=True,
                    p=qtof_probs
                )
            else:
                sampled = [None] * n_samples

            for ce in sampled:
                new_row = row.to_dict()
                new_row["COLLISION ENERGY"] = ce
                expanded_rows.append(new_row)

        expanded_df = pd.DataFrame(expanded_rows)

        if i == 0:
            expanded_df.to_csv(output_file, index=False)
        else:
            expanded_df.to_csv(output_file, index=False, header=False, mode="a")

        print(f"Chunk {i+1} processed")

    print("All done.")
    return output_file


if __name__ == "__main__":
    sample_collision_energy(PROVIDED_FILE, NIST_FILE, OUTPUT_FILE, CHUNK_SIZE, N_SAMPLES)