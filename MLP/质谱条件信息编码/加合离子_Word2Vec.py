import pandas as pd
import numpy as np
import re
from gensim.models import Word2Vec

VECTOR_SIZE = 16

def tokenize_precursor(p):
    p = p.replace("[", "").replace("]", "")
    return re.findall(r'M|\+|\-|\d+|[A-Z][a-z]*', p)

def encode_precursor_type_df(df):
    ID_COL = df.columns[0]
    ids = df[ID_COL].tolist()
    precursors = df["PRECURSOR TYPE"].astype(str).tolist()

    sentences = [tokenize_precursor(p) for p in precursors]

    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=10,
        min_count=1,
        sg=0,
        workers=4
    )

    embeddings = []
    for p in precursors:
        vec = np.zeros(VECTOR_SIZE)
        for t in tokenize_precursor(p):
            vec += model.wv[t]
        embeddings.append(vec)

    out_df = pd.DataFrame(
        embeddings,
        columns=[f"feat_precursor_{i}" for i in range(VECTOR_SIZE)]
    )
    out_df.insert(0, "ID", ids)
    return out_df


if __name__ == "__main__":
    input_path = r"D:\实验\数据\Nist数据\MLP\前5000条\output_5000.xlsx"
    output_path = r"D:\实验\数据\Nist数据\MLP\前5000条\precursor_type_5000.xlsx"

    df = pd.read_excel(input_path)
    out_df = encode_precursor_type_df(df)
    out_df.to_excel(output_path, index=False)

    print("✅ precursor type Word2Vec 编码完成")
    print(out_df.head())
