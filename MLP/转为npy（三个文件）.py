import os
import numpy as np
import pandas as pd

# =========================
# 路径配置
# =========================
CSV_ROOT = r"D:\实验\数据\Nist数据\MLP\1008403训练MLP模型"
NPY_ROOT = r"D:\实验\数据\Nist数据\MLP\1008403训练MLP模型"
os.makedirs(NPY_ROOT, exist_ok=True)

FP_CSV = os.path.join(CSV_ROOT, "morgan_fp_1002_full.csv")
MS_CSV = os.path.join(CSV_ROOT, "all_features.csv")
Y_CSV  = os.path.join(CSV_ROOT, "dreams_embedding_all.csv")

# =========================
# 通用 CSV → NPY 函数
# =========================
def csv_to_npy(csv_path, out_path, drop_id=True, dtype=np.float32):
    print(f"Loading {csv_path}")
    df = pd.read_csv(csv_path)

    # 保证按 ID 排序
    df = df.sort_values("ID")

    ids = df["ID"].to_numpy()

    if drop_id:
        data = df.drop(columns=["ID"]).to_numpy(dtype=dtype)
    else:
        data = df.to_numpy(dtype=dtype)

    np.save(out_path, data)
    return ids, data.shape


# =========================
# 转换三个文件
# =========================
ids_fp, fp_shape = csv_to_npy(FP_CSV, os.path.join(NPY_ROOT, "fp.npy"))
ids_ms, ms_shape = csv_to_npy(MS_CSV, os.path.join(NPY_ROOT, "ms.npy"))
ids_y,  y_shape  = csv_to_npy(Y_CSV,  os.path.join(NPY_ROOT, "y.npy"))

# =========================
# 一致性校验
# =========================
assert np.array_equal(ids_fp, ids_ms), "FP 和 MS 的 ID 不一致"
assert np.array_equal(ids_fp, ids_y),  "FP 和 Y 的 ID 不一致"

np.save(os.path.join(NPY_ROOT, "ids.npy"), ids_fp)

print("✅ 转换完成")
print("fp.npy:", fp_shape)
print("ms.npy:", ms_shape)
print("y.npy :", y_shape)
