import pandas as pd

MAX_EXACT_MASS = 1500.0


def encode_exact_mass_df(df):
    ID_COL = df.columns[0]

    mass = pd.to_numeric(df["EXACT MASS"], errors="coerce")
    mass_norm = (mass / MAX_EXACT_MASS).clip(lower=0, upper=1)

    out_df = pd.DataFrame({
        "ID": df[ID_COL],
        "feat_exact_mass": mass_norm
    })
    return out_df


if __name__ == "__main__":
    input_path = r"D:\实验\数据\Nist数据\MLP\前5000条\output_5000.xlsx"
    output_path = r"D:\实验\数据\Nist数据\MLP\前5000条\exact_mass_encoded_5000.xlsx"

    df = pd.read_excel(input_path)
    out_df = encode_exact_mass_df(df)
    out_df.to_excel(output_path, index=False)

    print("✅ exact mass 编码完成")
    print(out_df.head())