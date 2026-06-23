import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from encylogic import config

INPUT_DIR = str(config.VERILOG_DIR)
OUTPUT_DIR = str(config.BLIF_DIR)
YOSYS_CMD = config.YOSYS_BIN

MAX_WORKERS = multiprocessing.cpu_count()

def run_yosys_single_task(file_info):
    """
    这是单个进程执行的函数
    """
    input_path, output_path, filename = file_info

    yosys_script = f"read_verilog {input_path}; synth -auto-top; write_blif {output_path}"

    try:
        subprocess.run(
            [YOSYS_CMD, "-q", "-p", yosys_script],
            capture_output=True,
            text=True,
            check=True
        )
        return (True, filename, None)
    except subprocess.CalledProcessError as e:
        return (False, filename, e.stderr)
    except Exception as e:
        return (False, filename, str(e))

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 错误: 输入文件夹 '{INPUT_DIR}' 不存在。")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    v_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.v')]
    total_files = len(v_files)

    if total_files == 0:
        print("⚠️  没有找到 .v 文件。")
        return

    tasks = []
    for f in v_files:
        in_path = os.path.join(INPUT_DIR, f)
        out_path = os.path.join(OUTPUT_DIR, os.path.splitext(f)[0] + ".blif")
        tasks.append((in_path, out_path, f))

    print(f"🚀 开始并行转换 (使用 {MAX_WORKERS} 个 CPU 核心)...")
    print(f"📦 总文件数: {total_files}")
    print("-" * 40)

    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_yosys_single_task, task) for task in tasks]

        for i, future in enumerate(as_completed(futures)):
            is_success, filename, error_msg = future.result()

            if is_success:
                print(f"[{i+1}/{total_files}] ✅ {filename}")
                success_count += 1
            else:
                print(f"[{i+1}/{total_files}] ❌ {filename} 失败")
                fail_count += 1

    print("-" * 40)
    print(f"🎉 并行处理完成! 成功: {success_count}, 失败: {fail_count}")
    print(f"📂 结果保存在: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
