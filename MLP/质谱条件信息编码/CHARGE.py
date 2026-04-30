import pandas as pd

MAX_ABS_CHARGE = 4.0

def encode_charge_df(df):
    ID_COL = df.columns[0]

    def encode_charge(x):
        if pd.isna(x):
            return 0.0
        try:
            return float(x) / MAX_ABS_CHARGE
        except:
            return 0.0

    out_df = pd.DataFrame({
        "ID": df[ID_COL],
        "feat_charge": df["CHARGE"].apply(encode_charge)
    })
    return out_df


if __name__ == "__main__":
    input_path = r"D:\实验\数据\Nist数据\MLP\前5000条\output_5000.xlsx"
    output_path = r"D:\实验\数据\Nist数据\MLP\前5000条\charge_encoded_5000.xlsx"

    df = pd.read_excel(input_path)
    out_df = encode_charge_df(df)
    out_df.to_excel(output_path, index=False)

    print("✅ charge 编码完成")
    print(out_df.head())
