import pandas as pd

def encode_ion_mode_df(df):
    ID_COL = df.columns[0]

    def encode_ionmode(v):
        if pd.isna(v):
            return -1
        s = str(v).strip().lower()
        if s in ["p", "positive"]:
            return 1
        elif s in ["n", "negative"]:
            return 0
        else:
            return -1

    out_df = pd.DataFrame({
        "ID": df[ID_COL],
        "feat_ion_mode": df["ION MODE"].apply(encode_ionmode)
    })
    return out_df


if __name__ == "__main__":
    input_path = r"D:\实验\数据\Nist数据\MLP\前5000条\output_5000.xlsx"
    output_path = r"D:\实验\数据\Nist数据\MLP\前5000条\ion_mode_encoded_5000.xlsx"

    df = pd.read_excel(input_path)
    out_df = encode_ion_mode_df(df)
    out_df.to_excel(output_path, index=False)

    print("✅ ion mode 编码完成")
    print(out_df.head())
