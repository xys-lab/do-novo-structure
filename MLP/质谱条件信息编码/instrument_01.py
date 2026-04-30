import pandas as pd

def encode_instrument_df(df):
    ID_COL = df.columns[0]

    def encode_instrument(instr):
        if pd.isna(instr):
            return -1
        s = str(instr).lower()
        if "orbitrap" in s:
            return 1
        elif "qtof" in s:
            return 0
        else:
            return -1

    out_df = pd.DataFrame({
        "ID": df[ID_COL],
        "feat_instrument": df["INSTRUMENT"].apply(encode_instrument)
    })
    return out_df


if __name__ == "__main__":
    input_path = r"D:\实验\数据\Nist数据\MLP\前5000条\output_5000.xlsx"
    output_path = r"D:\实验\数据\Nist数据\MLP\前5000条\instrument_encoded_5000.xlsx"

    df = pd.read_excel(input_path)
    out_df = encode_instrument_df(df)
    out_df.to_excel(output_path, index=False)

    print("✅ instrument 编码完成")
    print(out_df.head())
