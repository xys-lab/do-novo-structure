import pandas as pd

MAX_PRECURSOR_MZ = 1500.0


def encode_precursor_mz_df(df):
    ID_COL = df.columns[0]

    mz = pd.to_numeric(df["PRECURSOR M/Z"], errors="coerce")
    mz_norm = (mz / MAX_PRECURSOR_MZ).clip(lower=0, upper=1)

    out_df = pd.DataFrame({
        "ID": df[ID_COL],
        "feat_precursor_mz": mz_norm
    })
    return out_df


if __name__ == "__main__":
    input_path = r"D:\实验\数据\Nist数据\MLP\前5000条\output_5000.xlsx"
    output_path = r"D:\实验\数据\Nist数据\MLP\前5000条\precursor_mz_encoded_5000.xlsx"

    df = pd.read_excel(input_path)
    out_df = encode_precursor_mz_df(df)
    out_df.to_excel(output_path, index=False)

    print("✅ precursor m/z 编码完成")
    print(out_df.head())