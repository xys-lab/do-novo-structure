import os
import re
import pandas as pd
from gensim.models import Word2Vec

VECTOR_SIZE = 16
SEED = 42

# -----------------------------
# 正则表达式
# -----------------------------
RE_CHARGE = re.compile(r'(\d*)([+-])$')  # 尾部电荷
RE_MULTIMER = re.compile(r'^\d+M$')  # 2M, 3M
RE_NUMBER = re.compile(r'^\d+(\.\d+)?$')  # 纯数值
RE_ELEMENT = re.compile(r'([A-Z][a-z]?)(\d*)')  # 元素+数量
RE_ADDUCT_PART = re.compile(r'([+-])([^+-\]]+)')  # +H -CH2O2

# 元素列表
ELEMENT_LIST = [
    "H","C","O","N","S","P","F","Cl","Br","I",
    "Na","K","Ca","Fe","Li","Mg","Al","Si","Ti",
    "Zn","As","B"
]

# ✅ 特殊 token（已扩展）
SPECIAL_TOKENS = ["M", "2M", "3M", "Cat", "i"]

# -----------------------------
# 单个 token 规范化
# -----------------------------
def normalize_token(token):
    token = token.strip()
    if not token:
        return "UNK"

    # ✅ 数值统一
    if RE_NUMBER.fullmatch(token):
        return "NUM"

    # ✅ 特殊 token 优先
    if token in SPECIAL_TOKENS:
        return token

    # multimer
    if RE_MULTIMER.fullmatch(token):
        return token

    # 元素解析
    units = RE_ELEMENT.findall(token)
    if units:
        token_parts = []
        for e, n in units:
            if e in ELEMENT_LIST:
                token_parts.append(f"{e}{n}" if n else e)
            else:
                token_parts.append(f"SPECIAL_{e}{n}" if n else f"SPECIAL_{e}")
        return "".join(token_parts)

    # 普通字母
    if re.fullmatch(r'[A-Za-z]+', token):
        return token

    return f"SPECIAL_{token}"

# -----------------------------
# tokenize precursor
# -----------------------------
def tokenize_precursor(precursor):
    if pd.isna(precursor):
        return ["UNK"]

    s = str(precursor).strip()
    if not s:
        return ["UNK"]

    s = s.replace(" ", "")

    # 提取电荷
    charge_token = None
    m_charge = RE_CHARGE.search(s)
    if m_charge:
        charge_num = int(m_charge.group(1)) if m_charge.group(1) else 1
        charge_sign = "P" if m_charge.group(2) == "+" else "N"
        charge_token = f"CHARGE_{charge_num}{charge_sign}"
        s = s[:m_charge.start()]

    # 去括号
    s = s.strip("[]")

    tokens = []
    # ⭐ 新增：处理开头的 M / 2M / 3M
    m_multimer = re.match(r'^(\d+M|M)(?![a-z])', s)

    if m_multimer:
        first = m_multimer.group(1)

        if first in SPECIAL_TOKENS or RE_MULTIMER.fullmatch(first):
            tokens.append(first)
            s = s[len(first):]  # 从字符串中移除

    # ✅ 纯数值
    if RE_NUMBER.fullmatch(s):
        tokens.append("NUM")
        if charge_token:
            tokens.append(charge_token)
        return tokens

    # 拆 + / -
    parts = RE_ADDUCT_PART.findall(s)

    if parts:
        for sign, formula in parts:
            tokens.append(sign)

            # ✅ 特殊 token（优先）
            if formula in SPECIAL_TOKENS:
                tokens.append(formula)
                continue

            # ✅ multimer
            if RE_MULTIMER.fullmatch(formula):
                tokens.append(formula)
                continue

            # 元素解析
            elems = RE_ELEMENT.findall(formula)

            if elems:
                for e, n in elems:
                    if e in ELEMENT_LIST:
                        tokens.append(f"{e}{n}" if n else e)
                    else:
                        tokens.append(f"SPECIAL_{e}{n}" if n else f"SPECIAL_{e}")
            else:
                tokens.append(f"SPECIAL_{formula}")

    else:
        # 整体判断
        if s in SPECIAL_TOKENS:
            tokens.append(s)

        elif RE_MULTIMER.fullmatch(s):
            tokens.append(s)

        else:
            elems = RE_ELEMENT.findall(s)

            if elems:
                for e, n in elems:
                    if e in ELEMENT_LIST:
                        tokens.append(f"{e}{n}" if n else e)
                    else:
                        tokens.append(f"SPECIAL_{e}{n}" if n else f"SPECIAL_{e}")
            else:
                tokens.append(f"SPECIAL_{s}")

    if charge_token:
        tokens.append(charge_token)

    return tokens if tokens else ["UNK"]

# -----------------------------
# 训练 Word2Vec
# -----------------------------
def train_precursor_w2v(input_csv, model_path, text_col="PRECURSOR TYPE", chunksize=50000):
    sentences = []

    reader = pd.read_csv(input_csv, chunksize=chunksize)

    for chunk_id, df in enumerate(reader, start=1):
        print(f"读取 chunk {chunk_id}, 行数 {len(df)}")

        vals = df[text_col].tolist()

        for v in vals:
            tokens = tokenize_precursor(v)
            sentences.append(tokens)

    print(f"总句子数: {len(sentences)}")

    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=10,
        min_count=1,
        sg=0,
        workers=4,
        seed=SEED,
        epochs=20
    )

    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    model.save(model_path)
    print(f"模型已保存: {model_path}")

# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    INPUT_CSV = r"D:\实验\数据\Nist数据\MLP\nist20数据集（1008403clean）.csv"
    MODEL_PATH = r"D:\实验\数据\Nist数据\MLP\1008403训练MLP模型\质谱条件信息编码（加合物形式Word2Vec模型）\precursor_type_word2vec.model"

    train_precursor_w2v(INPUT_CSV, MODEL_PATH)