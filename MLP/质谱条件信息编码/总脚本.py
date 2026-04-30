import os
import pandas as pd

from instrument_01 import encode_instrument_df
from ion_mode_01 import encode_ion_mode_df
from PRECURSOR_质荷比_归一化 import encode_precursor_mz_df
from precursor_type_fixed_encoder import encode_precursor_type_df
from CHARGE import encode_charge_df
from COLLISION_ENERGY_统一CE import encode_ce_df
from EXACT_MASS_归一化 import encode_exact_mass_df


# ============================================================
# 配置
# ============================================================

INPUT_CSV = r"F:\从头生成任务1\训练\cleaned_DSSTox_canonica_随机质谱\batch_000001_provided_with_all.csv"
OUTPUT_CSV = r"F:\从头生成任务1\训练\all_features_000001.csv"
PROBLEM_CSV = r"F:\从头生成任务1\训练\problem_all_features_000001.csv"

CHUNK_SIZE = 50000

encode_funcs = [
    encode_instrument_df,
    encode_ion_mode_df,
    encode_precursor_mz_df,
    encode_precursor_type_df,
    encode_charge_df,
    encode_ce_df,
    encode_exact_mass_df,
]


# ============================================================
# 工具函数
# ============================================================

def build_empty_like(base_id, template_columns):
    """
    构造与模板列一致的空表：
    - 第一列固定为 ID
    - 其余特征列全部填 0
    - 行数与 base_id 完全一致
    - 索引强制重置为 0 ~ n-1
    """
    out = pd.DataFrame({"ID": base_id}).reset_index(drop=True)

    for col in template_columns:
        if col != "ID":
            out[col] = 0

    return out.reset_index(drop=True)


def get_template_from_sample(func, sample_df):
    """
    用一个小样本探测该编码模块输出列。
    若失败，则返回 ['ID']。
    """
    try:
        tmp = func(sample_df.copy())
        tmp = tmp.reset_index(drop=True)

        cols = list(tmp.columns)

        if "ID" not in cols:
            cols = ["ID"] + [c for c in cols if c != "ID"]

        return cols

    except Exception as e:
        print(f"⚠️ 模板探测失败: {func.__name__} -> {e}")
        return ["ID"]


def flush_problem_rows(problem_rows_buffer, problem_csv, write_header):
    """
    将问题记录追加写入 CSV，避免问题很多时长期占用内存。
    """
    if not problem_rows_buffer:
        return write_header

    pd.DataFrame(
        problem_rows_buffer,
        columns=["chunk_id", "ID", "module", "error_type"]
    ).to_csv(
        problem_csv,
        mode="a",
        header=write_header,
        index=False
    )

    return False


# ============================================================
# 删除旧文件
# ============================================================

if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)

if os.path.exists(PROBLEM_CSV):
    os.remove(PROBLEM_CSV)


# ============================================================
# 预探测各模块输出列
# ============================================================

print("开始预探测各编码模块输出列...")

sample_df = pd.read_csv(INPUT_CSV, nrows=min(100, CHUNK_SIZE))
sample_df = sample_df.reset_index(drop=True)

module_templates = {}

for func in encode_funcs:
    print(f"探测 {func.__name__}")
    cols = get_template_from_sample(func, sample_df)
    module_templates[func.__name__] = cols
    print(f"{func.__name__} 输出列数: {len(cols)}")
    print(f"{func.__name__} 输出列: {cols}")

print("\n预探测完成。")


# ============================================================
# 主流程：分块流式处理
# ============================================================

reader = pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE)

total_in = 0
total_out = 0
chunk_id = 0

problem_rows_buffer = []
problem_header_needed = True

# 只新增这一行：控制“向量列说明”只打印一次
mapping_printed = False

for df in reader:
    chunk_id += 1

    # 关键修复1：每个 chunk 一开始就重置索引
    df = df.reset_index(drop=True)

    print(f"\n处理 chunk {chunk_id} 行数 {len(df)}")

    total_in += len(df)

    ID_COL = df.columns[0]
    base_id = df[ID_COL].values

    feature_dfs = []

    # ===================================
    # 逐模块编码
    # ===================================

    for func in encode_funcs:
        func_name = func.__name__
        template_cols = module_templates.get(func_name, ["ID"])

        print(f"执行 {func_name}")

        try:
            out_df = func(df.copy())

            # 关键修复2：模块输出后立刻重置索引
            out_df = out_df.reset_index(drop=True)

            # 必须包含 ID
            if "ID" not in out_df.columns:
                raise ValueError("输出结果缺少 ID 列")

            # 补齐模板列
            for col in template_cols:
                if col not in out_df.columns:
                    out_df[col] = 0

            # 只保留模板列，且顺序固定
            out_df = out_df[template_cols]

            # 再次重置索引，防止补列/切列后索引异常
            out_df = out_df.reset_index(drop=True)

            # 行数检查
            if len(out_df) != len(df):
                print(f"❌ 行数异常: {func_name} -> 输入 {len(df)}，输出 {len(out_df)}")

                for id_ in base_id:
                    problem_rows_buffer.append(
                        (chunk_id, id_, func_name, "ROW_COUNT_MISMATCH")
                    )

                out_df = build_empty_like(base_id, template_cols)

            else:
                # 顺序检查：必须逐行一致
                try:
                    same_order = (out_df["ID"].values == base_id).all()
                except Exception:
                    same_order = False

                if not same_order:
                    print(f"❌ 顺序异常: {func_name}")

                    for id_ in base_id:
                        problem_rows_buffer.append(
                            (chunk_id, id_, func_name, "ORDER_MISMATCH")
                        )

                    out_df = build_empty_like(base_id, template_cols)

            # 缺失值填 0 + 强制转 float
            for col in out_df.columns:
                if col != "ID":
                    out_df[col] = pd.to_numeric(out_df[col], errors="coerce").fillna(0).astype("float32")

            # 最后再统一一次索引
            out_df = out_df.reset_index(drop=True)

        except Exception as e:
            print(f"❌ 执行失败: {func_name} -> {e}")

            for id_ in base_id:
                problem_rows_buffer.append(
                    (chunk_id, id_, func_name, f"EXCEPTION: {e}")
                )

            out_df = build_empty_like(base_id, template_cols)

        # 再保险：最终必须和输入同样行数
        if len(out_df) != len(df):
            print(f"❌ 严重异常: {func_name} 替代后行数仍不一致，强制重建")
            out_df = build_empty_like(base_id, template_cols)

        # 再保险：最终 ID 必须逐行一致
        try:
            final_same = (out_df["ID"].values == base_id).all()
        except Exception:
            final_same = False

        if not final_same:
            print(f"❌ 严重异常: {func_name} 替代后ID仍不一致，强制重建")
            out_df = build_empty_like(base_id, template_cols)

        # 关键修复3：加入列表前再次统一索引
        out_df = out_df.reset_index(drop=True)
        feature_dfs.append(out_df)

    # ===================================
    # 拼接前统一所有特征表索引
    # ===================================

    feature_dfs = [x.reset_index(drop=True) for x in feature_dfs]

    # ===================================
    # 横向拼接
    # ===================================

    final_df = feature_dfs[0].copy().reset_index(drop=True)

    for d in feature_dfs[1:]:
        final_df = pd.concat(
            [
                final_df.reset_index(drop=True),
                d.drop(columns=["ID"]).reset_index(drop=True)
            ],
            axis=1
        )

    # 关键修复4：拼接完成后再次重置索引
    final_df = final_df.reset_index(drop=True)

    # chunk 级检查
    if len(final_df) != len(df):
        raise ValueError(
            f"chunk {chunk_id} 最终输出行数异常: 输入 {len(df)}，输出 {len(final_df)}"
        )

    try:
        final_same = (final_df["ID"].values == base_id).all()
    except Exception:
        final_same = False

    if not final_same:
        raise ValueError(f"chunk {chunk_id} 最终输出ID顺序异常")

    total_out += len(final_df)

    # ===================================
    # 只修改这部分：打印一次列说明 + 重命名为 ms_0, ms_1, ...
    # ===================================

    original_feature_cols = list(final_df.columns[1:])

    if not mapping_printed:
        print("\n===== 向量列说明（仅打印一次） =====")
        for i, col in enumerate(original_feature_cols):
            print(f"ms_{i} <- {col}")
        print("==================================\n")
        mapping_printed = True

    final_df.columns = ["ID"] + [f"ms_{i}" for i in range(len(original_feature_cols))]
    for col in final_df.columns:
        if col != "ID":
            final_df[col] = pd.to_numeric(final_df[col], errors="coerce").fillna(0).astype("float32")

    # ===================================
    # 写出当前 chunk
    # ===================================

    final_df.to_csv(
        OUTPUT_CSV,
        mode="a",
        header=not os.path.exists(OUTPUT_CSV),
        index=False
    )

    print(f"chunk {chunk_id} 写入完成 | 输入 {len(df)} 行 -> 输出 {len(final_df)} 行")

    # ===================================
    # 问题日志分批落盘
    # ===================================

    if len(problem_rows_buffer) >= 10000:
        problem_header_needed = flush_problem_rows(
            problem_rows_buffer,
            PROBLEM_CSV,
            problem_header_needed
        )
        problem_rows_buffer = []

# 处理剩余问题记录
if problem_rows_buffer:
    problem_header_needed = flush_problem_rows(
        problem_rows_buffer,
        PROBLEM_CSV,
        problem_header_needed
    )
    problem_rows_buffer = []

# ============================================================
# 最终统计
# ============================================================

print("\n======================")
print(f"输入总行数: {total_in}")
print(f"输出总行数: {total_out}")
print("======================")

if total_in == total_out:
    print("✅ 输入输出总行数一致")
else:
    print("❌ 输入输出总行数不一致")

if os.path.exists(PROBLEM_CSV):
    print(f"问题记录文件已生成: {PROBLEM_CSV}")
else:
    print("未发现问题记录。")