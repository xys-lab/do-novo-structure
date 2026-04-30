import os
import gc
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from multiprocessing import Pool
from collections import Counter

########################################
# 配置
########################################

INPUT_CSV = r"F:\从头生成任务1\训练\cleaned_DSSTox_canonica_随机质谱\batch_000001_provided_with_all.csv"
OUTPUT_CSV = r"F:\从头生成任务1\训练\morgan_fp_1002_000001.csv"
LOG_CSV    = r"F:\从头生成任务1\训练\morgan_fp_failed_000001.csv"

SMILES_COL = "canonical_smiles"

FP_BITS = 1002
CHUNK_SIZE = 50000
N_WORKERS = 3

########################################
# 初始化指纹生成器
########################################

generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=FP_BITS
)

########################################
# SMILES → FP
########################################

def smiles_to_fp(args):
    mol_id, smiles = args

    if not isinstance(smiles, str) or smiles.strip() == "":
        return mol_id, smiles, None, "SMILES为空"

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return mol_id, smiles, None, "RDKit无法解析"

        fp = generator.GetFingerprint(mol)

        arr = np.empty(FP_BITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)

        return mol_id, smiles, arr, None

    except Exception as e:
        return mol_id, smiles, None, str(e)

########################################
# 主程序
########################################

if __name__ == "__main__":

    print("\n====== Morgan指纹生成（流式写入CSV） ======\n")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # 统计总行数
    total_rows = 0
    for chunk in pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE):
        total_rows += len(chunk)

    print(f"真实数据条数: {total_rows:,}")

    ok, fail = 0, 0
    fail_records = []
    fail_counter = Counter()

    first_chunk = True

    pool = Pool(N_WORKERS)

    ########################################
    # 分块处理
    ########################################
    for chunk_idx, df in enumerate(
        pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE)
    ):

        print(f"\n--- 处理 chunk {chunk_idx+1} ---")

        id_col = df.columns[0]

        ids = df[id_col].tolist()
        smiles_list = df[SMILES_COL].tolist()

        tasks = list(zip(ids, smiles_list))

        ########################################
        # 并行计算
        ########################################
        results = pool.map(smiles_to_fp, tasks)

        ########################################
        # 构建输出（无 SMILES）
        ########################################
        records = []

        for mol_id, smiles, fp, reason in results:

            if fp is None:
                # 失败：NaN占位
                records.append([mol_id] + [np.nan] * FP_BITS)

                fail += 1
                fail_counter[reason] += 1
                fail_records.append([mol_id, smiles, reason])
            else:
                # 成功：转 float32（避免后面类型混乱）
                records.append([mol_id] + fp.astype(np.float32).tolist())
                ok += 1

        ########################################
        # 写入 CSV（关键）
        ########################################
        out_df = pd.DataFrame(
            records,
            columns=[
                id_col,
                *[f"fp_{i}" for i in range(FP_BITS)]
            ]
        )

        out_df.to_csv(
            OUTPUT_CSV,
            mode="w" if first_chunk else "a",
            index=False,
            header=first_chunk
        )

        first_chunk = False

        print(f"累计成功: {ok:,} | 失败: {fail:,}")

        del df, records, out_df
        gc.collect()

    pool.close()
    pool.join()

    ########################################
    # 保存失败日志
    ########################################
    if fail_records:
        pd.DataFrame(
            fail_records,
            columns=[id_col, SMILES_COL, "fail_reason"]
        ).to_csv(LOG_CSV, index=False)

    ########################################
    # 输出统计
    ########################################
    print("\n====== 完成 ======")
    print(f"成功: {ok:,}")
    print(f"失败: {fail:,}")

    for k, v in fail_counter.items():
        print(f"{k}: {v:,}")