from multiprocessing import freeze_support
import os
import pandas as pd

from top_adduct10 import assign_top_adducts
from instrument import expand_instrument
from collision_energy import sample_collision_energy
from remaining_ms import compute_remaining_ms_info

# =========================================================
# 配置区
# =========================================================
INPUT_FILE = r"D:\实验\数据\DSSTox_Feb_2024\cleaned_DSSTox_canonical.csv"
NIST_FILE = r"D:\实验\数据\Nist数据\MLP\nist20数据集（1008403clean）.csv"
OUTPUT_DIR = r"F:\从头生成任务1\训练\cleaned_DSSTox_canonica_随机质谱"

ADDUCT_PROB_FILE = r"D:\实验\数据\Nist数据\MLP\随机添加10条质谱条件信息\adduct_probabilities.csv"
ADDUCT_RULE_FILE = r"D:\实验\数据\Nist数据\MLP\随机添加10条质谱条件信息\adduct_structure_rules.csv"

# 前5份：每份原始输入5000行，保留中间文件
SMALL_BATCH_SIZE = 5000
SMALL_BATCH_COUNT = 5

# 后续：每份原始输入50000行，只保留最终文件
LARGE_BATCH_SIZE = 50000

# 是否删除大批次的中间文件
DELETE_INTERMEDIATE_FOR_LARGE_BATCH = True


# =========================================================
# 工具函数
# =========================================================
def count_csv_rows(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        total = sum(1 for _ in f) - 1
    return max(total, 0)


def print_file_rows(title, file_path):
    if os.path.exists(file_path):
        rows = count_csv_rows(file_path)
        print(f"{title}: {rows} 行 -> {file_path}")
    else:
        print(f"{title}: 文件不存在 -> {file_path}")


def safe_remove(file_path):
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"已删除中间文件: {file_path}")
        except Exception as e:
            print(f"删除失败: {file_path}, 原因: {e}")


def process_one_batch(
    batch_idx,
    batch_input_file,
    keep_intermediate,
):
    """
    处理单个批次：
    batch_input_file -> top10 -> instrument -> CE -> provided_with_all
    """
    batch_prefix = f"batch_{batch_idx:06d}"

    topk_file = os.path.join(OUTPUT_DIR, f"{batch_prefix}_top10_adducts.csv")
    instrument_file = os.path.join(OUTPUT_DIR, f"{batch_prefix}_with_instrument.csv")
    ce_file = os.path.join(OUTPUT_DIR, f"{batch_prefix}_provided_with_CE.csv")
    final_file = os.path.join(OUTPUT_DIR, f"{batch_prefix}_provided_with_all.csv")
    error_log_file = os.path.join(OUTPUT_DIR, f"{batch_prefix}_adduct_errors.csv")

    print("\n" + "=" * 70)
    print(f"开始处理第 {batch_idx} 份")
    print(f"输入文件: {batch_input_file}")
    print(f"是否保留中间文件: {keep_intermediate}")
    print("=" * 70)

    # 如果最终文件已经存在，直接跳过
    if os.path.exists(final_file):
        print_file_rows("该份最终文件已存在，跳过", final_file)
        return

    # Step 1
    if not os.path.exists(topk_file):
        print("\n[Step 1] 预测 Top-K 加合物")
        assign_top_adducts(batch_input_file, ADDUCT_PROB_FILE, ADDUCT_RULE_FILE, topk_file)
        print_file_rows("Step 1 输出数据量", topk_file)
    else:
        print_file_rows("Step 1 输出已存在，跳过", topk_file)

    # Step 2
    if not os.path.exists(instrument_file):
        print("\n[Step 2] 扩展仪器类型")
        expand_instrument(topk_file, instrument_file)
        print_file_rows("Step 2 输出数据量", instrument_file)
    else:
        print_file_rows("Step 2 输出已存在，跳过", instrument_file)

    # Step 3
    if not os.path.exists(ce_file):
        print("\n[Step 3] 采样碰撞能")
        sample_collision_energy(instrument_file, NIST_FILE, ce_file)
        print_file_rows("Step 3 输出数据量", ce_file)
    else:
        print_file_rows("Step 3 输出已存在，跳过", ce_file)

    # Step 4
    if not os.path.exists(final_file):
        print("\n[Step 4] 计算其余质谱信息")
        compute_remaining_ms_info(ce_file, final_file, error_log_file)
        print_file_rows("Step 4 输出数据量", final_file)
    else:
        print_file_rows("Step 4 输出已存在，跳过", final_file)

    print(f"\n第 {batch_idx} 份处理完成")
    print_file_rows("最终文件", final_file)

    # 大批次不保留中间文件
    if not keep_intermediate and DELETE_INTERMEDIATE_FOR_LARGE_BATCH:
        print("\n开始删除该份中间文件...")
        safe_remove(topk_file)
        safe_remove(instrument_file)
        safe_remove(ce_file)

    print("-" * 70)


# =========================================================
# 主流程
# =========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_rows = count_csv_rows(INPUT_FILE)
    print("=" * 70)
    print("开始运行总流程（自定义分份模式）")
    print(f"原始输入文件: {INPUT_FILE}")
    print(f"原始总行数: {total_rows}")
    print(f"前 {SMALL_BATCH_COUNT} 份: 每份 {SMALL_BATCH_SIZE} 行，保留中间文件")
    print(f"之后每份: {LARGE_BATCH_SIZE} 行，只保留最终 provided_with_all.csv")
    print("=" * 70)

    # 读取整个输入文件的流式 reader
    reader = pd.read_csv(INPUT_FILE, chunksize=SMALL_BATCH_SIZE)

    batch_idx = 0
    rows_processed = 0

    # -----------------------------------------------------
    # 第一阶段：前5份，每份5000
    # -----------------------------------------------------
    for _ in range(SMALL_BATCH_COUNT):
        try:
            chunk = next(reader)
        except StopIteration:
            break

        batch_idx += 1
        rows_processed += len(chunk)

        batch_input_file = os.path.join(OUTPUT_DIR, f"batch_{batch_idx:06d}_input.csv")
        if not os.path.exists(batch_input_file):
            chunk.to_csv(batch_input_file, index=False)
            print_file_rows(f"第 {batch_idx} 份输入文件已保存", batch_input_file)
        else:
            print_file_rows(f"第 {batch_idx} 份输入文件已存在", batch_input_file)

        process_one_batch(
            batch_idx=batch_idx,
            batch_input_file=batch_input_file,
            keep_intermediate=True,
        )

    # -----------------------------------------------------
    # 第二阶段：后续合并成每份50000
    # 因为前面 reader 还是按5000读，所以这里每10个 chunk 合并一次
    # -----------------------------------------------------
    pending_chunks = []

    for chunk in reader:
        pending_chunks.append(chunk)

        # 50000 / 5000 = 10 个小块合并成一个大份
        if len(pending_chunks) == (LARGE_BATCH_SIZE // SMALL_BATCH_SIZE):
            batch_idx += 1
            merged_chunk = pd.concat(pending_chunks, ignore_index=True)
            rows_processed += len(merged_chunk)

            batch_input_file = os.path.join(OUTPUT_DIR, f"batch_{batch_idx:06d}_input.csv")
            if not os.path.exists(batch_input_file):
                merged_chunk.to_csv(batch_input_file, index=False)
                print_file_rows(f"第 {batch_idx} 份输入文件已保存", batch_input_file)
            else:
                print_file_rows(f"第 {batch_idx} 份输入文件已存在", batch_input_file)

            process_one_batch(
                batch_idx=batch_idx,
                batch_input_file=batch_input_file,
                keep_intermediate=False,
            )

            pending_chunks = []

    # 处理最后不足50000的一份
    if len(pending_chunks) > 0:
        batch_idx += 1
        merged_chunk = pd.concat(pending_chunks, ignore_index=True)
        rows_processed += len(merged_chunk)

        batch_input_file = os.path.join(OUTPUT_DIR, f"batch_{batch_idx:06d}_input.csv")
        if not os.path.exists(batch_input_file):
            merged_chunk.to_csv(batch_input_file, index=False)
            print_file_rows(f"第 {batch_idx} 份输入文件已保存", batch_input_file)
        else:
            print_file_rows(f"第 {batch_idx} 份输入文件已存在", batch_input_file)

        process_one_batch(
            batch_idx=batch_idx,
            batch_input_file=batch_input_file,
            keep_intermediate=False,
        )

    print("\n" + "=" * 70)
    print("所有分份处理完成")
    print(f"总共处理份数: {batch_idx}")
    print("=" * 70)


if __name__ == "__main__":
    freeze_support()
    main()