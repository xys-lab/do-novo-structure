import pandas as pd
import re

def encode_ce_df(df):
    ID_COL = df.columns[0]

    pattern_nce = re.compile(r"NCE\s*=\s*(\d+\.?\d*)\s*%")
    pattern_ev = re.compile(r"(\d+\.?\d*)\s*eV")

    numeric_vals, ev_vals = [], []

    for v in df["COLLISION ENERGY"].dropna():
        s = str(v)
        if pattern_ev.search(s):
            ev_vals.append(float(pattern_ev.search(s).group(1)))
        elif s.replace(".", "", 1).isdigit():
            numeric_vals.append(float(s))

    max_numeric = max(numeric_vals) if numeric_vals else 1.0
    max_ev = max(ev_vals) if ev_vals else None

    def encode_ce(v):
        if pd.isna(v):
            return -1
        s = str(v)

        m = pattern_nce.search(s)
        if m:
            return float(m.group(1)) / 100.0

        m = pattern_ev.search(s)
        if m and max_ev:
            return float(m.group(1)) / max_ev

        if s.replace(".", "", 1).isdigit():
            return float(s) / max_numeric

        return -1

    out_df = pd.DataFrame({
        "ID": df[ID_COL],
        "CE_strength": df["COLLISION ENERGY"].apply(encode_ce)
    })
    return out_df


if __name__ == "__main__":
    input_path = r"D:\实验\数据\Nist数据\MLP\前5000条\output_5000.xlsx"
    output_path = r"D:\实验\数据\Nist数据\MLP\前5000条\collision_energy_encoded_5000.xlsx"

    df = pd.read_excel(input_path)
    out_df = encode_ce_df(df)
    out_df.to_excel(output_path, index=False)

    print("✅ collision energy 编码完成")
    print(out_df.head())
