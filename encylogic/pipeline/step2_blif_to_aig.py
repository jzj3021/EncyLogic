import os
import subprocess
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from encylogic import config

try:
    from encylogic.actionspace import ACTION_SPACE
except ImportError:
    print("❌ 错误: 找不到 actionspace.py 文件或 ACTION_SPACE 变量。")
    exit(1)

INPUT_DIR = str(config.BLIF_DIR)
OUTPUT_DIR = str(config.AIG_DIR)
ABC_CMD = config.ABC_BIN

VARIANTS_PER_DESIGN = 8

def generate_aig_task(file_info):
    """
    单个进程执行的任务：将 1 个 BLIF 转换为多个结构各异的 AIG
    使用 GIA 空间 (& space) 进行加速优化
    """
    blif_path, output_subdir, filename_base = file_info

    generated_files = []
    errors = []

    for i in range(VARIANTS_PER_DESIGN):
        out_name = f"{filename_base}_{i}.aig"
        out_path = os.path.join(output_subdir, out_name)

        cmds = []

        cmds.append(f"read_blif {blif_path}")
        cmds.append("strash")

        if i > 0:
            cmds.append("&get")

            num_ops = random.randint(5, 15)
            random_seq = [random.choice(ACTION_SPACE) for _ in range(num_ops)]
            cmds.extend(random_seq)

            cmds.append("&st")
            cmds.append("&put")

        cmds.append(f"write_aiger {out_path}")
        cmds.append("quit")

        abc_script = "; ".join(cmds)

        try:
            res = subprocess.run(
                [ABC_CMD, "-c", abc_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120
            )

            if res.returncode == 0 and os.path.exists(out_path):
                generated_files.append(out_name)
            else:
                err_msg = res.stderr.strip() if res.stderr else "Unknown Error"
                if "unknown command" in res.stdout:
                     err_msg = "Command Error (Check ABC version)"
                errors.append(f"Fail {out_name}: {err_msg}")

        except subprocess.TimeoutExpired:
            errors.append(f"Timeout {out_name}")
        except Exception as e:
            errors.append(f"Err {out_name}: {str(e)}")

    return generated_files, errors

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 输入目录 {INPUT_DIR} 不存在。")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 已创建输出目录: {OUTPUT_DIR}")

    blif_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.blif')]
    total_files = len(blif_files)

    if total_files == 0:
        print("⚠️  目录中未找到 .blif 文件。")
        return

    num_cpus = 28
    total_aigs_expected = total_files * VARIANTS_PER_DESIGN

    print("=" * 50)
    print(f"🚀 EncyLogic 数据集生成器 (GIA Space Mode)")
    print(f"📄 输入电路数: {total_files}")
    print(f"🔀 每个电路变体: {VARIANTS_PER_DESIGN}")
    print(f"📦 预计生成总数: {total_aigs_expected}")
    print(f"🔥 并行核心数: {num_cpus}")
    print("=" * 50)

    tasks = []
    for f in blif_files:
        path = os.path.join(INPUT_DIR, f)
        base_name = os.path.splitext(f)[0]
        tasks.append((path, OUTPUT_DIR, base_name))

    completed_aigs = 0

    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = [executor.submit(generate_aig_task, t) for t in tasks]

        for i, future in enumerate(as_completed(futures)):
            gen_files, errs = future.result()
            completed_aigs += len(gen_files)

            progress = (i + 1) / total_files * 100
            status_symbol = "✅" if not errs else "⚠️ "

            print(f"[{progress:5.1f}%] {status_symbol} 任务 {i+1}/{total_files} 完成 | 本次生成: {len(gen_files)}/{VARIANTS_PER_DESIGN}")

            if errs:
                print(f"        └── 错误示例: {', '.join(errs[:1])}")

    print("=" * 50)
    print(f"🎉 全部处理完成！")
    print(f"📊 实际生成 AIG 文件数: {completed_aigs}")
    print(f"📂 存储路径: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
