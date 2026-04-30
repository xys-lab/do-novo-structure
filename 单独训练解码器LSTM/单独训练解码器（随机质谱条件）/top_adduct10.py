import os
import csv
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from collections import defaultdict
from multiprocessing import Pool, freeze_support

# ==============================
# 文件路径
# ==============================
DSSTOX_FILE = r"D:\实验\数据\DSSTox_Feb_2024\dsstox_cleaned.csv"
ADDUCT_PROB_FILE = r"D:\实验\数据\Nist数据\MLP\随机添加10条质谱条件信息\adduct_probabilities.csv"
ADDUCT_RULE_FILE = r"D:\实验\数据\Nist数据\MLP\随机添加10条质谱条件信息\adduct_structure_rules.csv"

OUTPUT_DIR = r"F:\从头生成任务1\训练"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_TOP10_FILE = os.path.join(OUTPUT_DIR, "top10_adducts.csv")

# ==============================
# 关闭 RDKit 冗余日志
# ==============================
RDLogger.DisableLog("rdApp.*")

# ==============================
# SMARTS结构规则
# ==============================
STRUCT_SMARTS = {
    "carboxylic_acid": "C(=O)[OH]",
    "phenol": "c[OX2H]",
    "sulfonic_acid": "S(=O)(=O)[OH]",
    "phosphate": "P(=O)(O)(O)",
    "thiol": "[SX2H]",
    "amine": "[NX3;H2,H1,H0]",
    "amide": "C(=O)N",
    "aniline_like_N": "c[NX3;H2,H1,H0]",
    "six_membered_aromatic_N": "[n;r6]",
    "five_membered_aromatic_N": "[n;r5]",
    "hydroxyl": "[OX2H]",
    "ether": "[OD2]([#6])[#6]",
    "ester": "C(=O)O[#6]",
    "carbonyl": "[CX3]=[OX1]",
    "ketone": "[#6][CX3](=O)[#6]",
    "aldehyde": "[CX3H1](=O)[#6,H]",
    "nitrile": "[CX2]#N",
    "aromatic_ring": "a",
    "halogen": "[F,Cl,Br,I]"
}

PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in STRUCT_SMARTS.items()}

# ==============================
# 参数
# ==============================
ALPHA = 0.7
BETA = 0.3
TOP_K = 10

CHUNK_SIZE = 50000
CPU_CORES = 2
POOL_CHUNKSIZE = 200

FLOAT_FORMAT = "%.6f"

# ==============================
# 子进程全局变量
# ==============================
WORKER_ADDUCT_PRIOR = None
WORKER_RULE_DICT = None
WORKER_CANDIDATE_ADDUCTS = None

# ==============================
# 结构匹配
# ==============================
def detect_structures(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    matched = {
        name
        for name, pattern in PATTERNS.items()
        if pattern is not None and mol.HasSubstructMatch(pattern)
    }
    return matched

# ==============================
# 读取参考文件
# ==============================
def load_reference_files():
    adduct_df = pd.read_csv(ADDUCT_PROB_FILE)
    adduct_prior = dict(zip(adduct_df["adduct"], adduct_df["base_probability"]))

    rule_df = pd.read_csv(ADDUCT_RULE_FILE)
    rule_dict = defaultdict(list)

    for _, row in rule_df.iterrows():
        adduct = row["adduct"]
        structure = row["structure"]
        prob = float(row["structure_probability"])
        rule_dict[adduct].append((structure, prob))

    candidate_adducts = list(adduct_prior.keys())
    return adduct_prior, rule_dict, candidate_adducts

# ==============================
# worker 初始化
# ==============================
def init_worker(adduct_prior, rule_dict, candidate_adducts):
    global WORKER_ADDUCT_PRIOR, WORKER_RULE_DICT, WORKER_CANDIDATE_ADDUCTS
    WORKER_ADDUCT_PRIOR = adduct_prior
    WORKER_RULE_DICT = rule_dict
    WORKER_CANDIDATE_ADDUCTS = candidate_adducts

# ==============================
# 计算单个分子的总概率
# ==============================
def score_one_smiles(smiles):
    matched_structures = detect_structures(smiles)
    if matched_structures is None:
        return None

    scores = {}
    for adduct in WORKER_CANDIDATE_ADDUCTS:
        base_prob = WORKER_ADDUCT_PRIOR.get(adduct, 0.0)

        struct_prob = 0.0
        for struct, prob in WORKER_RULE_DICT.get(adduct, []):
            if struct in matched_structures:
                struct_prob += prob

        total_prob = ALPHA * base_prob + BETA * struct_prob
        scores[adduct] = total_prob

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    return sorted_scores

# ==============================
# 多进程处理单条分子
# 这里明确按列名取值，保证 canonical_smiles / inchikey / exact_mass 对应关系不乱
# ==============================
def process_row(row):
    smiles = row["canonical_smiles"]
    inchikey = row["inchikey"]
    exact_mass = row["exact_mass"]

    if pd.isna(smiles):
        return None, inchikey, exact_mass, None

    smiles = str(smiles)
    inchikey = "" if pd.isna(inchikey) else str(inchikey)
    exact_mass = "" if pd.isna(exact_mass) else str(exact_mass)

    top_adducts = score_one_smiles(smiles)
    return smiles, inchikey, exact_mass, top_adducts

# ==============================
# 统计总行数
# ==============================
def count_total_rows(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        total = sum(1 for _ in f) - 1
    return max(total, 0)

# ==============================
# 对外接口
# ==============================
def assign_top_adducts(
    dsstox_file,
    adduct_prob_file,
    adduct_rule_file,
    output_file,
    alpha=0.7,
    beta=0.3,
    top_k=10,
    chunk_size=50000,
    cpu_cores=2,
    pool_chunksize=200,
):
    global DSSTOX_FILE, ADDUCT_PROB_FILE, ADDUCT_RULE_FILE, OUTPUT_TOP10_FILE
    global ALPHA, BETA, TOP_K, CHUNK_SIZE, CPU_CORES, POOL_CHUNKSIZE

    DSSTOX_FILE = dsstox_file
    ADDUCT_PROB_FILE = adduct_prob_file
    ADDUCT_RULE_FILE = adduct_rule_file
    OUTPUT_TOP10_FILE = output_file

    ALPHA = alpha
    BETA = beta
    TOP_K = top_k
    CHUNK_SIZE = chunk_size
    CPU_CORES = cpu_cores
    POOL_CHUNKSIZE = pool_chunksize

    main()
    return OUTPUT_TOP10_FILE

# ==============================
# 主流程
# ==============================
def main():
    print("===================================")
    print("Loading reference probability files")
    print("===================================")

    adduct_prior, rule_dict, candidate_adducts = load_reference_files()

    print(f"Adduct types: {len(candidate_adducts)}")
    print(f"CPU_CORES: {CPU_CORES}")
    print(f"CSV CHUNK_SIZE: {CHUNK_SIZE}")
    print(f"POOL_CHUNKSIZE: {POOL_CHUNKSIZE}")
    print("===================================")

    total_rows = count_total_rows(DSSTOX_FILE)

    required_cols = ["canonical_smiles", "inchikey", "exact_mass"]

    with open(OUTPUT_TOP10_FILE, "w", newline="", encoding="utf-8-sig") as f_out:
        fieldnames = ["canonical_smiles", "inchikey", "exact_mass", "PRECURSOR TYPE"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        with Pool(
            processes=CPU_CORES,
            initializer=init_worker,
            initargs=(adduct_prior, rule_dict, candidate_adducts)
        ) as pool:

            with tqdm(total=total_rows, desc="Processing DSSTox") as pbar:
                for chunk in pd.read_csv(
                    DSSTOX_FILE,
                    usecols=required_cols,
                    chunksize=CHUNK_SIZE
                ):
                    # 显式固定列顺序，避免 CSV 原始列顺序导致错位
                    chunk = chunk[["canonical_smiles", "inchikey", "exact_mass"]]

                    # 按列名转字典，而不是按位置转 tuple
                    rows = chunk.to_dict("records")

                    for smiles, inchikey, exact_mass, top_adducts in pool.imap(
                        process_row,
                        rows,
                        chunksize=POOL_CHUNKSIZE
                    ):
                        if top_adducts is None:
                            for _ in range(TOP_K):
                                writer.writerow({
                                    "canonical_smiles": smiles,
                                    "inchikey": inchikey,
                                    "exact_mass": exact_mass,
                                    "PRECURSOR TYPE": None,
                                })
                        else:
                            for adduct, total_prob in top_adducts:
                                writer.writerow({
                                    "canonical_smiles": smiles,
                                    "inchikey": inchikey,
                                    "exact_mass": exact_mass,
                                    "PRECURSOR TYPE": adduct,
                                })

                        pbar.update(1)

                    f_out.flush()
                    del chunk
                    del rows

    print("===================================")
    print("Top 10 adducts per molecule saved to:")
    print(OUTPUT_TOP10_FILE)
    print("===================================")

if __name__ == "__main__":
    freeze_support()
    main()