import os
import pandas as pd
from collections import defaultdict, Counter
from rdkit import Chem
from rdkit import RDLogger

# ==============================
# 全局配置
# ==============================
RDLogger.DisableLog("rdApp.*")

# ==============================
# 输入文件
# ==============================
NIST_FILE = r"D:\实验\数据\Nist数据\MLP\nist20数据集（1008403clean）.csv"

# ==============================
# 输出目录
# ==============================
OUTPUT_DIR = r"D:\实验\数据\Nist数据\MLP\随机10条质谱条件信息"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# 输出文件
# ==============================
ADDUCT_PROB_FILE = os.path.join(OUTPUT_DIR, "adduct_probabilities.csv")
ADDUCT_RULE_FILE = os.path.join(OUTPUT_DIR, "adduct_structure_rules.csv")
FILTERED_FILE = os.path.join(OUTPUT_DIR, "nist_filtered.csv")  # 用于中间筛选

# ==============================
# 列名设置
# ==============================
NIST_SMILES_COL = "SMILES"
NIST_ADDUCT_COL = "PRECURSOR TYPE"

# ==============================
# 参数设置
# ==============================
CHUNK_SIZE = 50000
P_STRUCTURE_GIVEN_ADDUCT_THRESHOLD = 0.05
MIN_COUNT_ADDUCT = 100
FLOAT_FORMAT = "%.6f"

# ==============================
# SMARTS结构定义
# ==============================
STRUCT_SMARTS = {
    # 酸性相关
    "carboxylic_acid": "C(=O)[OH]",
    "phenol": "c[OX2H]",
    "sulfonic_acid": "S(=O)(=O)[OH]",
    "phosphate": "P(=O)(O)(O)",
    "thiol": "[SX2H]",

    # 可质子化相关
    "amine": "[NX3;H2,H1,H0]",
    "amide": "C(=O)N",
    "aniline_like_N": "c[NX3;H2,H1,H0]",
    "six_membered_aromatic_N": "[n;r6]",
    "five_membered_aromatic_N": "[n;r5]",

    # 配位 / 金属加合相关
    "hydroxyl": "[OX2H]",
    "ether": "[OD2]([#6])[#6]",
    "ester": "C(=O)O[#6]",
    "carbonyl": "[CX3]=[OX1]",
    "ketone": "[#6][CX3](=O)[#6]",
    "aldehyde": "[CX3H1](=O)[#6,H]",
    "nitrile": "[CX2]#N",

    # 辅助结构
    "aromatic_ring": "a",
    "halogen": "[F,Cl,Br,I]"
}

PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in STRUCT_SMARTS.items()}

# ==============================
# 识别结构
# 返回当前分子命中的结构集合
# ==============================
def detect_structures(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    matched = [name for name, pattern in PATTERNS.items() if pattern is not None and mol.HasSubstructMatch(pattern)]
    return set(matched)

# ==============================
# Step1：统计全部加合物基础概率
# 输出 adduct_probabilities.csv
# ==============================
def step1_adduct_frequency():
    print("===================================")
    print("Step 1: Building P(adduct) for all adduct types")
    print("===================================")

    counts = Counter()
    total_rows = 0

    for chunk_id, chunk in enumerate(pd.read_csv(NIST_FILE, usecols=[NIST_ADDUCT_COL], chunksize=CHUNK_SIZE), start=1):
        print(f"Processing chunk {chunk_id} ...")
        chunk = chunk.dropna(subset=[NIST_ADDUCT_COL])
        adduct_list = chunk[NIST_ADDUCT_COL].astype(str).tolist()
        counts.update(adduct_list)
        total_rows += len(adduct_list)

    rows = []
    for adduct, count in counts.items():
        prob = float(count) / float(total_rows) if total_rows > 0 else 0.0
        rows.append({
            "adduct": adduct,
            "count_adduct": float(count),
            "base_probability": float(prob)
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["base_probability", "count_adduct"], ascending=[False, False]).reset_index(drop=True)

    df.to_csv(ADDUCT_PROB_FILE, index=False, encoding="utf-8-sig", float_format=FLOAT_FORMAT)
    print("Total unique adduct types:", len(df))
    print("Saved:", ADDUCT_PROB_FILE)

# ==============================
# Step2：统计 P(structure | adduct)
# 输出 adduct_structure_rules.csv
# 新增 count_adduct, count_adduct_structure
# 筛选 P>0.05 且 count_adduct>=100
# ==============================
def step2_adduct_structure_rules():
    print("===================================")
    print("Step 2: Building P(structure | adduct)")
    print("===================================")

    adduct_counts = defaultdict(int)
    structure_adduct_counts = defaultdict(int)

    total_valid_smiles = 0
    total_invalid_smiles = 0

    for chunk_id, chunk in enumerate(pd.read_csv(NIST_FILE, usecols=[NIST_SMILES_COL, NIST_ADDUCT_COL], chunksize=CHUNK_SIZE), start=1):
        print(f"Processing chunk {chunk_id} ...")
        chunk = chunk.dropna(subset=[NIST_SMILES_COL, NIST_ADDUCT_COL])

        for smiles, adduct in zip(chunk[NIST_SMILES_COL].astype(str), chunk[NIST_ADDUCT_COL].astype(str)):
            matched = detect_structures(smiles)
            if matched is None:
                total_invalid_smiles += 1
                continue

            total_valid_smiles += 1
            adduct_counts[adduct] += 1

            for structure in matched:
                structure_adduct_counts[(adduct, structure)] += 1

    # 生成规则表
    rows = []
    for (adduct, structure), count_as in structure_adduct_counts.items():
        count_a = adduct_counts[adduct]
        prob = float(count_as) / float(count_a) if count_a > 0 else 0.0

        # 筛选 P>0.05 且 count_adduct>=100
        if prob > P_STRUCTURE_GIVEN_ADDUCT_THRESHOLD and count_a >= 100:
            rows.append({
                "adduct": adduct,
                "structure": structure,
                "count_adduct": float(count_a),
                "count_adduct_structure": float(count_as),
                "structure_probability": float(prob)
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["adduct", "structure_probability", "count_adduct_structure"], ascending=[True, False, False]).reset_index(drop=True)

    df.to_csv(ADDUCT_RULE_FILE, index=False, encoding="utf-8-sig", float_format=FLOAT_FORMAT)
    print("Valid SMILES:", total_valid_smiles)
    print("Invalid SMILES:", total_invalid_smiles)
    print("Rules kept after filtering:", len(df))
    print("Saved:", ADDUCT_RULE_FILE)


def build_adduct_probability_files(nist_file, output_dir, chunk_size=50000, p_threshold=0.05, min_count_adduct=100):
    global NIST_FILE, OUTPUT_DIR, ADDUCT_PROB_FILE, ADDUCT_RULE_FILE
    global CHUNK_SIZE, P_STRUCTURE_GIVEN_ADDUCT_THRESHOLD, MIN_COUNT_ADDUCT

    NIST_FILE = nist_file
    OUTPUT_DIR = output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ADDUCT_PROB_FILE = os.path.join(OUTPUT_DIR, "adduct_probabilities.csv")
    ADDUCT_RULE_FILE = os.path.join(OUTPUT_DIR, "adduct_structure_rules.csv")

    CHUNK_SIZE = chunk_size
    P_STRUCTURE_GIVEN_ADDUCT_THRESHOLD = p_threshold
    MIN_COUNT_ADDUCT = min_count_adduct

    step1_adduct_frequency()
    step2_adduct_structure_rules()
    return ADDUCT_PROB_FILE, ADDUCT_RULE_FILE
# ==============================
# 主函数
# ==============================
def main():
    step1_adduct_frequency()
    step2_adduct_structure_rules()
    print("===================================")
    print("All probability files completed")
    print("===================================")

if __name__ == "__main__":
    main()