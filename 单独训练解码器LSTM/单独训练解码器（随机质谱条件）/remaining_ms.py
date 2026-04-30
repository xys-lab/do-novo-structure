import os
import pandas as pd
import re
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
from tqdm import tqdm

# -----------------------------
# 全局参数
# -----------------------------
element_mass = {
    "H": 1.0078250319,
    "C": 12.0,
    "N": 14.00307400425,
    "O": 15.9949146193,
    "S": 31.9720711735,
    "P": 30.9737619977,
    "F": 18.9984031621,
    "Cl": 34.96885269,
    "Br": 78.9183376,
    "I": 126.904473,
    "Na": 22.989769282,
    "K": 38.963706485,
    "Li": 7.016003434,
    "Fe": 55.93493554,
    "Ca":39.962590851
}

electron_mass = 0.0005486

# 明确缩写 -> 普通分子式
adduct_map = {
    "FA": "CHO2",
    "NH4": "H4N",
    "H2O": "H2O"
}

error_log_file = r"D:\实验\数据\DSSTox_Feb_2024\试验小样本5000\adduct_errors.csv"


# -----------------------------
# 日志函数
# -----------------------------
def write_log(message: str):
    with open(error_log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")


# -----------------------------
# 分子式相关函数
# -----------------------------
def expand_formula_tokens(formula: str):
    """
    把普通分子式拆成元素+计数
    例如:
        HCl   -> [('H', 1), ('Cl', 1)]
        C2H6O -> [('C', 2), ('H', 6), ('O', 1)]
    """
    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    result = []
    for elem, cnt in tokens:
        cnt = int(cnt) if cnt else 1
        result.append((elem, cnt))
    return result


def is_formula_like(text: str):
    """
    判断一个字符串是否像普通分子式
    例如:
        HCl, HCN, NO, CHNS, C2H6O3, Na, K, Fe
    """
    return bool(re.fullmatch(r'(?:[A-Z][a-z]?\d*)+', text))


def compute_formula_mass(formula: str, row_index: int):
    """
    计算普通分子式的质量
    如果有未知元素，记录 warning，并跳过未知元素，只累加已知元素
    """
    mass = 0.0
    tokens = expand_formula_tokens(formula)

    if not tokens:
        write_log(f"[ADDUCT MASS WARNING] row {row_index}, invalid formula: {formula}")
        return 0.0

    for elem, cnt in tokens:
        if elem in element_mass:
            mass += element_mass[elem] * cnt
        else:
            write_log(f"[ADDUCT MASS WARNING] row {row_index}, unknown element: {elem}")
    return mass


# -----------------------------
# 精确质量
# -----------------------------
def compute_exact_mass_safe(args):
    smiles, row_index = args
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("无法解析 SMILES")
        return float(rdMolDescriptors.CalcExactMolWt(mol))
    except Exception as e:
        write_log(f"[EXACT MASS ERROR] row {row_index}, SMILES: {smiles}, Error: {e}")
        return None


# -----------------------------
# 加合物字符串切分
# -----------------------------
def split_adduct_pieces(inside: str):
    """
    把方括号内部拆成带符号的片段
    例如:
        M+H-HCl+2i -> [('+', 'M'), ('+', 'H'), ('-', 'HCl'), ('+', '2i')]
        2M+Na      -> [('+', '2M'), ('+', 'Na')]
    """
    inside = inside.replace(" ", "")
    if not inside:
        return []

    if inside[0] not in "+-":
        inside = "+" + inside

    return re.findall(r'([+-])([^+-]+)', inside)


# -----------------------------
# 加合物解析
# -----------------------------
def parse_adduct_safe(args):
    adduct_str, row_index = args

    if pd.isna(adduct_str) or not isinstance(adduct_str, str):
        write_log(f"[ADDUCT PARSE WARNING] row {row_index}, invalid or empty adduct: {adduct_str}")
        return None, None, None, None

    try:
        match = re.search(r'\[(.*?)\]', adduct_str)
        if not match:
            write_log(f"[ADDUCT PARSE WARNING] row {row_index}, missing brackets: {adduct_str}")
            return None, None, None, None

        adduct_inside = match.group(1).strip()

        charge_match = re.search(r'\](\d*)([+-])', adduct_str)
        charge = 1
        if charge_match:
            charge_number = int(charge_match.group(1)) if charge_match.group(1) else 1
            charge_sign = 1 if charge_match.group(2) == '+' else -1
            charge = charge_sign * charge_number

        parts = split_adduct_pieces(adduct_inside)

        adduct_mass = 0.0
        m_multiplier = 0

        for sign, piece in parts:
            sign_factor = 1.0 if sign == '+' else -1.0
            piece = piece.strip()

            m_num = re.fullmatch(r'(\d*)(M)', piece)
            if m_num:
                mult = int(m_num.group(1)) if m_num.group(1) else 1
                m_multiplier += sign_factor * mult
                continue

            general_num = re.fullmatch(r'(\d+)(.+)', piece)
            if general_num:
                prefix_count = int(general_num.group(1))
                core = general_num.group(2)
            else:
                prefix_count = 1
                core = piece

            # ==============================
            # 新规则
            # ==============================

            # Cat 当作 M
            if core == "Cat":
                m_multiplier += sign_factor * prefix_count
                continue

            # i 直接跳过
            if core == "i":
                continue

            # ==============================

            if core in adduct_map:
                formula = adduct_map[core]
                piece_mass = compute_formula_mass(formula, row_index)
                adduct_mass += sign_factor * prefix_count * piece_mass
                continue

            if is_formula_like(core):
                piece_mass = compute_formula_mass(core, row_index)
                adduct_mass += sign_factor * prefix_count * piece_mass
                continue

            write_log(f"[ADDUCT MASS WARNING] row {row_index}, unknown token skipped: {core}")

        if m_multiplier == 0:
            m_multiplier = 1

        if charge > 0:
            adduct_mass -= abs(charge) * electron_mass
        else:
            adduct_mass += abs(charge) * electron_mass

        ion_mode = 'P' if charge > 0 else 'N'

        return float(adduct_mass), int(charge), ion_mode, float(m_multiplier)

    except Exception as e:
        write_log(f"[ADDUCT PARSE ERROR] row {row_index}, adduct: {adduct_str}, Error: {e}")
        return None, None, None, None


# -----------------------------
# 计算前体 m/z
# -----------------------------
def compute_precursor_mz_safe(args):
    exact_mass, adduct_mass, charge, m_multiplier, row_index = args
    try:
        if exact_mass is None or adduct_mass is None or charge is None or m_multiplier is None:
            return None
        return float((exact_mass * m_multiplier + adduct_mass) / abs(charge))
    except Exception as e:
        write_log(f"[PRECURSOR M/Z ERROR] row {row_index}, Error: {e}")
        return None


def compute_remaining_ms_info(
    input_file,
    output_file,
    error_log_path,
    chunk_size=10000,
    num_cores=3,
    max_rows=10_000_000,
):
    global error_log_file
    error_log_file = error_log_path

    first_chunk = True
    rows_processed = 0
    global_id = 0

    if os.path.exists(error_log_file):
        os.remove(error_log_file)

    for chunk_index, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        if rows_processed >= max_rows:
            break

        print(f"开始处理 chunk {chunk_index + 1}")

        if rows_processed + len(chunk) > max_rows:
            chunk = chunk.iloc[:max_rows - rows_processed]

        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            chunk['EXACT MASS'] = list(tqdm(
                executor.map(
                    compute_exact_mass_safe,
                    [(smi, idx + rows_processed) for idx, smi in enumerate(chunk['canonical_smiles'])]
                ),
                total=len(chunk)
            ))

        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            parsed = list(tqdm(
                executor.map(
                    parse_adduct_safe,
                    [(adduct, idx + rows_processed) for idx, adduct in enumerate(chunk['PRECURSOR TYPE'])]
                ),
                total=len(chunk)
            ))

        parsed_safe = [p if p is not None else (None, None, None, None) for p in parsed]

        chunk['ADDUCT MASS'] = [p[0] for p in parsed_safe]
        chunk['CHARGE'] = [p[1] for p in parsed_safe]
        chunk['ION MODE'] = [p[2] for p in parsed_safe]
        chunk['M MULTIPLIER'] = [p[3] for p in parsed_safe]

        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            chunk['PRECURSOR M/Z'] = list(tqdm(
                executor.map(
                    compute_precursor_mz_safe,
                    [
                        (m, a, c, mm, idx + rows_processed)
                        for idx, (m, a, c, mm) in enumerate(
                            zip(
                                chunk['EXACT MASS'],
                                chunk['ADDUCT MASS'],
                                chunk['CHARGE'],
                                chunk['M MULTIPLIER']
                            )
                        )
                    ]
                ),
                total=len(chunk)
            ))

        float_cols = ['EXACT MASS', 'ADDUCT MASS', 'PRECURSOR M/Z', 'M MULTIPLIER']
        chunk['CHARGE'] = pd.to_numeric(chunk['CHARGE'], errors='coerce').astype('Int64')

        for col in float_cols:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce').astype(float)

        start_id = global_id + 1
        end_id = global_id + len(chunk)

        chunk.insert(0, "ID", range(start_id, end_id + 1))

        for col in ['EXACT MASS', 'ADDUCT MASS', 'PRECURSOR M/Z']:
            chunk[col] = chunk[col].round(6)

        chunk.to_csv(
            output_file,
            index=False,
            mode='w' if first_chunk else 'a',
            header=first_chunk,
            float_format='%.6f'
        )

        first_chunk = False
        rows_processed += len(chunk)
        global_id += len(chunk)
        print(f"完成 chunk {chunk_index + 1}")

    print("完成：已处理数据的一半")
    print("结果保存在", output_file)
    print("错误和警告记录在", error_log_file)
    return output_file


# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    freeze_support()

    input_file = r"D:\实验\数据\DSSTox_Feb_2024\试验小样本5000\provided_with_CE_5000.csv"
    output_file = r"D:\实验\数据\DSSTox_Feb_2024\试验小样本5000\provided_with_all_5000.csv"
    error_log_path = r"D:\实验\数据\DSSTox_Feb_2024\试验小样本5000\adduct_errors.csv"

    compute_remaining_ms_info(
        input_file=input_file,
        output_file=output_file,
        error_log_path=error_log_path,
        chunk_size=10000,
        num_cores=3,
        max_rows=10_000_000,
    )



