import math
import time
import re
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import pandas as pd
import requests
from tqdm import tqdm
import periodictable as pt


# =========================================================
# 配置区
# =========================================================
INPUT_FILES = [
    r"D:\实验\数据\DSSTox_Feb_2024\DSSToxDump3.xlsx",
    r"D:\实验\数据\DSSTox_Feb_2024\DSSToxDump4.xlsx",
    r"D:\实验\数据\DSSTox_Feb_2024\DSSToxDump5.xlsx",
    r"D:\实验\数据\DSSTox_Feb_2024\DSSToxDump6.xlsx",
    r"D:\实验\数据\DSSTox_Feb_2024\DSSToxDump7.xlsx",
    r"D:\实验\数据\DSSTox_Feb_2024\DSSToxDump8.xlsx",
    r"D:\实验\数据\DSSTox_Feb_2024\DSSToxDump9.xlsx",
    r"D:\实验\数据\DSSTox_Feb_2024\DSSToxDump10.xlsx",
    r"D:\实验\数据\DSSTox_Feb_2024\DSSToxDump11.xlsx",
    r"D:\实验\数据\DSSTox_Feb_2024\DSSToxDump12.xlsx",
    r"D:\实验\数据\DSSTox_Feb_2024\DSSToxDump13.xlsx",
    # 继续添加更多文件...
]

OUTPUT_DIR = Path(r"D:\实验\数据\DSSTox_Feb_2024\filled_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SMILES_COL = "SMILES"
INCHIKEY_COL = "INCHIKEY"
MONOISOTOPIC_MASS_COL = "MONOISOTOPIC_MASS"
FORMULA_COL = "MOLECULAR_FORMULA"

SLEEP_SECONDS = 0.2
TIMEOUT = 20
MAX_RETRIES = 3

# 是否在终端打印你选用的“主同位素质量”
PRINT_MAJOR_ISOTOPES = True


# =========================================================
# 主同位素选择（你现在已经是在手动选最常见同位素）
# =========================================================
MAJOR_ISOTOPES = {
    "H": 1,
    "C": 12,
    "N": 14,
    "O": 16,
    "F": 19,
    "P": 31,
    "S": 32,
    "Cl": 35,
    "Br": 79,
    "I": 127,
    "Si": 28,
    "B": 11,
    "Na": 23,
    "K": 39,
    "Ca": 40,
}


# =========================================================
# 全局缓存：避免重复查 PubChem
# =========================================================
SMILES_CACHE: Dict[str, Optional[Dict[str, str]]] = {}
INCHIKEY_CACHE: Dict[str, Optional[Dict[str, str]]] = {}


# =========================================================
# Session 复用连接
# =========================================================
SESSION = requests.Session()


# =========================================================
# 缺失判断
# =========================================================
def is_missing(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if str(x).strip() == "":
        return True
    return False


# =========================================================
# 打印当前选用的主同位素质量
# =========================================================
def print_major_isotope_masses():
    print("\n==============================")
    print("当前用于计算单同位素质量的主同位素")
    print("==============================")
    for elem, iso_num in MAJOR_ISOTOPES.items():
        iso_mass = getattr(pt, elem)[iso_num].mass
        print(f"{elem:>2}  [{iso_num}]  mass = {iso_mass}")
    print("==============================\n")


# =========================================================
# PubChem 请求（带重试）
# =========================================================
def request_json_with_retry(url: str) -> Optional[Dict]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = SESSION.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()

            # 非 200 也重试几次
            if attempt < MAX_RETRIES:
                time.sleep(SLEEP_SECONDS * attempt)
                continue
            return None

        except Exception:
            if attempt < MAX_RETRIES:
                time.sleep(SLEEP_SECONDS * attempt)
                continue
            return None


# =========================================================
# PubChem 查询
# =========================================================
def pubchem_get_by_smiles(smiles: str) -> Optional[Dict[str, str]]:
    smiles = smiles.strip()
    if smiles in SMILES_CACHE:
        return SMILES_CACHE[smiles]

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        f"{requests.utils.quote(smiles, safe='')}/property/"
        "InChIKey,CanonicalSMILES,MolecularFormula/JSON"
    )

    data = request_json_with_retry(url)
    if data is None:
        SMILES_CACHE[smiles] = None
        return None

    try:
        props = data["PropertyTable"]["Properties"][0]
        SMILES_CACHE[smiles] = props
        return props
    except Exception:
        SMILES_CACHE[smiles] = None
        return None


def pubchem_get_by_inchikey(inchikey: str) -> Optional[Dict[str, str]]:
    inchikey = inchikey.strip()
    if inchikey in INCHIKEY_CACHE:
        return INCHIKEY_CACHE[inchikey]

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/"
        f"{requests.utils.quote(inchikey, safe='')}/property/"
        "CanonicalSMILES,MolecularFormula/JSON"
    )

    data = request_json_with_retry(url)
    if data is None:
        INCHIKEY_CACHE[inchikey] = None
        return None

    try:
        props = data["PropertyTable"]["Properties"][0]
        INCHIKEY_CACHE[inchikey] = props
        return props
    except Exception:
        INCHIKEY_CACHE[inchikey] = None
        return None


# =========================================================
# 分子式 -> 元素计数
# =========================================================
def parse_formula(formula: str) -> Dict[str, int]:
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    out: Dict[str, int] = {}
    for elem, cnt in tokens:
        out[elem] = out.get(elem, 0) + (int(cnt) if cnt else 1)
    return out


# =========================================================
# 分子式 -> 单同位素质量（使用 periodictable）
# =========================================================
def calc_monoisotopic_mass_from_formula(formula: str) -> Optional[float]:
    if formula is None or str(formula).strip() == "":
        return None

    try:
        atom_counts = parse_formula(str(formula).strip())
        mass = 0.0

        for elem, cnt in atom_counts.items():
            if elem not in MAJOR_ISOTOPES:
                return None

            iso_num = MAJOR_ISOTOPES[elem]
            iso_mass = getattr(pt, elem)[iso_num].mass
            mass += iso_mass * cnt

        return mass

    except Exception:
        return None


# =========================================================
# 单文件处理
# =========================================================
def process_one_file(input_file: Path):
    # -----------------------------
    # 读取文件
    # -----------------------------
    if input_file.suffix.lower() == ".csv":
        df = pd.read_csv(input_file)
    else:
        df = pd.read_excel(input_file)

    # 保持原始顺序，仅清理列名空格
    df.columns = df.columns.astype(str).str.strip()

    required_cols = [SMILES_COL, INCHIKEY_COL, MONOISOTOPIC_MASS_COL]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{input_file.name} 缺失必要列: {col}")

    if FORMULA_COL not in df.columns:
        df[FORMULA_COL] = None

    stats = {
        "total_rows": len(df),
        "filled_smiles": 0,
        "filled_inchikey": 0,
        "filled_monoisotopic_mass": 0,
        "failed_fill_smiles": 0,
        "failed_fill_inchikey": 0,
        "failed_fill_monoisotopic_mass": 0,
        "all_three_still_missing_or_failed": 0,
    }

    # -----------------------------
    # 逐行补齐（严格保持顺序）
    # -----------------------------
    for idx in tqdm(range(len(df)), desc=f"Filling {input_file.name}"):
        smi = df.at[idx, SMILES_COL]
        ik = df.at[idx, INCHIKEY_COL]
        mono_mass = df.at[idx, MONOISOTOPIC_MASS_COL]
        formula = df.at[idx, FORMULA_COL]

        missing_smiles = is_missing(smi)
        missing_ik = is_missing(ik)
        missing_mass = is_missing(mono_mass)

        pubchem_data = None

        # -------------------------------------------------
        # 1) 缺 INCHIKEY，用 SMILES 查 PubChem
        # -------------------------------------------------
        if missing_ik and not missing_smiles:
            pubchem_data = pubchem_get_by_smiles(str(smi).strip())
            time.sleep(SLEEP_SECONDS)

            if pubchem_data and pubchem_data.get("InChIKey"):
                df.at[idx, INCHIKEY_COL] = pubchem_data["InChIKey"]
                ik = pubchem_data["InChIKey"]
                missing_ik = False
                stats["filled_inchikey"] += 1
            else:
                stats["failed_fill_inchikey"] += 1

            if pubchem_data and pubchem_data.get("MolecularFormula") and is_missing(formula):
                df.at[idx, FORMULA_COL] = pubchem_data["MolecularFormula"]
                formula = pubchem_data["MolecularFormula"]

        # -------------------------------------------------
        # 2) 缺 SMILES，用 INCHIKEY 查 PubChem
        # -------------------------------------------------
        if missing_smiles and not missing_ik:
            if pubchem_data is None:
                pubchem_data = pubchem_get_by_inchikey(str(ik).strip())
                time.sleep(SLEEP_SECONDS)

            if pubchem_data and pubchem_data.get("CanonicalSMILES"):
                df.at[idx, SMILES_COL] = pubchem_data["CanonicalSMILES"]
                smi = pubchem_data["CanonicalSMILES"]
                missing_smiles = False
                stats["filled_smiles"] += 1
            else:
                stats["failed_fill_smiles"] += 1

            if pubchem_data and pubchem_data.get("MolecularFormula") and is_missing(formula):
                df.at[idx, FORMULA_COL] = pubchem_data["MolecularFormula"]
                formula = pubchem_data["MolecularFormula"]

        # -------------------------------------------------
        # 3) 缺 MONOISOTOPIC_MASS，用公式计算
        # -------------------------------------------------
        if missing_mass:
            formula_to_use = None

            if pubchem_data and pubchem_data.get("MolecularFormula"):
                formula_to_use = pubchem_data["MolecularFormula"]
            elif not is_missing(formula):
                formula_to_use = formula

            calc_mass = calc_monoisotopic_mass_from_formula(formula_to_use) if formula_to_use else None

            if calc_mass is not None:
                df.at[idx, MONOISOTOPIC_MASS_COL] = calc_mass
                stats["filled_monoisotopic_mass"] += 1
            else:
                stats["failed_fill_monoisotopic_mass"] += 1

        # -------------------------------------------------
        # 4) 三项都仍然缺失/失败
        # -------------------------------------------------
        final_missing_smiles = is_missing(df.at[idx, SMILES_COL])
        final_missing_ik = is_missing(df.at[idx, INCHIKEY_COL])
        final_missing_mass = is_missing(df.at[idx, MONOISOTOPIC_MASS_COL])

        if final_missing_smiles and final_missing_ik and final_missing_mass:
            stats["all_three_still_missing_or_failed"] += 1

    # -----------------------------
    # 保存补齐后的文件（顺序不变）
    # -----------------------------
    output_file = OUTPUT_DIR / f"{input_file.stem}_filled{input_file.suffix}"

    if output_file.suffix.lower() == ".csv":
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(output_file, index=False)

    # -----------------------------
    # 终端打印统计
    # -----------------------------
    print("\n补齐完成")
    print(f"原始文件: {input_file}")
    print(f"补齐后文件: {output_file}")
    for k, v in stats.items():
        print(f"{k}: {v}")

    return stats


# =========================================================
# 主程序入口
# =========================================================
def main():
    if PRINT_MAJOR_ISOTOPES:
        print_major_isotope_masses()

    all_stats: List[Tuple[str, Dict[str, int]]] = []

    for file_path in INPUT_FILES:
        input_file = Path(file_path)
        if not input_file.exists():
            print(f"\n[跳过] 文件不存在: {input_file}")
            continue

        stats = process_one_file(input_file)
        all_stats.append((input_file.name, stats))

    # 全部文件汇总打印
    if len(all_stats) > 1:
        print("\n==============================")
        print("全部文件汇总")
        print("==============================")
        for file_name, stats in all_stats:
            print(f"\n文件: {file_name}")
            for k, v in stats.items():
                print(f"{k}: {v}")


if __name__ == "__main__":
    main()