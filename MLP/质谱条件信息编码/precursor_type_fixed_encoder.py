import pandas as pd
import numpy as np
from gensim.models import Word2Vec

from precursor_type_encoder import tokenize_precursor


MODEL_PATH = r"D:\实验\数据\Nist数据\MLP\1008403训练MLP模型\质谱条件信息编码（加合物形式Word2Vec模型）\precursor_type_word2vec.model"
ID_COLUMN = "ID"
PRECURSOR_COLUMN = "PRECURSOR TYPE"


class PrecursorEncoder:
    def __init__(self, model_path):
        self.model = Word2Vec.load(model_path)
        self.vector_size = self.model.vector_size
        self.unk_vector = np.mean(self.model.wv.vectors, axis=0)
        self.new_tokens = set()

    def encode(self, precursor):
        tokens = tokenize_precursor(precursor)

        vectors = []
        for tok in tokens:
            if tok in self.model.wv:
                vectors.append(self.model.wv[tok])
            else:
                vectors.append(self.unk_vector)
                self.new_tokens.add(tok)

        if not vectors:
            return self.unk_vector

        return np.mean(vectors, axis=0)

    def encode_batch(self, precursors):
        return np.array([self.encode(p) for p in precursors], dtype=np.float32)

    def get_new_tokens(self):
        return sorted(list(self.new_tokens))


# 全局只加载一次模型，避免每个 chunk 重复加载
_encoder = PrecursorEncoder(MODEL_PATH)


def encode_precursor_type_df(df):
    if ID_COLUMN not in df.columns:
        raise ValueError(f"列 {ID_COLUMN} 不存在")
    if PRECURSOR_COLUMN not in df.columns:
        raise ValueError(f"列 {PRECURSOR_COLUMN} 不存在")

    work_df = df.copy()
    work_df[PRECURSOR_COLUMN] = work_df[PRECURSOR_COLUMN].fillna("").astype(str)

    vectors = _encoder.encode_batch(work_df[PRECURSOR_COLUMN])

    vec_df = pd.DataFrame(
        vectors,
        columns=[f"precursor_vec_{i}" for i in range(_encoder.vector_size)]
    )

    out_df = pd.concat(
        [work_df[[ID_COLUMN]].reset_index(drop=True), vec_df.reset_index(drop=True)],
        axis=1
    )

    return out_df


def process_csv_stream(input_csv, output_csv, chunksize=50000):
    print("加载模型并处理文件...")

    reader = pd.read_csv(input_csv, chunksize=chunksize)
    first_chunk = True

    for chunk in reader:
        out_chunk = encode_precursor_type_df(chunk)

        if first_chunk:
            out_chunk.to_csv(output_csv, index=False)
            first_chunk = False
        else:
            out_chunk.to_csv(output_csv, mode="a", header=False, index=False)

    print("\n新出现的 token：")
    for t in _encoder.get_new_tokens():
        print(t)

    print("\n完成！")


if __name__ == "__main__":
    INPUT_CSV = r"D:\实验\数据\Nist数据\MLP\前5000条\nist20数据集（5000clean）.csv"
    OUTPUT_CSV = r"D:\实验\数据\Nist数据\MLP\前5000条\nist20数据集（5000clean）_pretype_only_id_vec.csv"

    process_csv_stream(INPUT_CSV, OUTPUT_CSV)