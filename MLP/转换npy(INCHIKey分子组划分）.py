import os
import numpy as np
import pandas as pd   # ← 你缺少的这一行

INCHI_CSV = r"D:\实验\数据\Nist数据\MLP\1008403\INCHIKey.csv"
NPY_ROOT  = r"D:\实验\数据\Nist数据\MLP\1008403"

df = pd.read_csv(INCHI_CSV)
df = df.sort_values("ID")

ids_inchi = df["ID"].to_numpy()
inchikey_array = df["INCHIKEY"].to_numpy()

ids_existing = np.load(os.path.join(NPY_ROOT, "ids.npy"))

assert np.array_equal(ids_inchi, ids_existing), "ID 不一致！"

np.save(os.path.join(NPY_ROOT, "inchikey.npy"), inchikey_array)

print("✅ InChIKey 转换完成")