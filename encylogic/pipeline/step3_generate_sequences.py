import os
import random
import subprocess
import re
import json
import time
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from encylogic import config

INPUT_DIR = str(config.AIG_DIR)
OUTPUT_DATA_FILE = str(config.DATA_DIR / "angledata.jsonl")
SEQ_LENGTH = 25
NUM_SEQUENCES_PER_DESIGN = 15000
KEEP_TOP_K = 100
MAX_WORKERS = 64

def get_abc_command():
    if os.path.exists(config.ABC_BIN): return config.ABC_BIN
    return config.ABC_BIN

try:
    from encylogic.actionspace import ACTION_SPACE
except ImportError:
    print("❌ 错误: 找不到 actionspace.py 文件或 ACTION_SPACE 变量。")
    exit(1)

ABC_CMD = get_abc_command()

def generate_safe_sequence(action_space, length):
    seq = []
    for _ in range(length):
        op = random.choice(action_space)
        seq.append(op)

    return seq

def run_abc_sequence(aig_path, sequence, debug=False):
    """
    [GIA 适配版] 增加调试信息输出
    """
    ops_str = " ; ".join(sequence)
    script = f"read {aig_path}; strash; &get; {ops_str}; &put; print_stats;"

    try:
        res = subprocess.run(
            [ABC_CMD, "-c", script],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=60
        )

        if res.returncode == 0:
            output = res.stdout
            nd_match = re.search(r'nd\s*=\s*(\d+)', output)
            lev_match = re.search(r'lev\s*=\s*(\d+)', output)

            if nd_match and lev_match:
                return int(nd_match.group(1)), int(lev_match.group(1))
            else:
                if debug: print(f"⚠️ [解析失败] 输出中未找到 nd/lev: {output.strip()[:100]}...")
                return None, None
        else:
            if debug: print(f"⚠️ [ABC报错] Cmd: {ops_str[:20]}... \n错误信息: {res.stderr.strip()}")
            return None, None

    except subprocess.TimeoutExpired:
        if debug: print(f"⏰ [超时] 电路处理超时: {aig_path}")
        return None, None
    except Exception as e:
        if debug: print(f"❌ [Python异常] {e}")
        return None, None

def process_design(aig_file):
    design_name = os.path.basename(aig_file).replace(".aig", "")

    initial_nodes, initial_levels = run_abc_sequence(aig_file, [], debug=True)
    if initial_nodes is None:
        print(f"❌ {design_name}: 初始读取失败 (可能是坏文件或ABC无法解析)")
        return []

    candidates = []
    rng = random.Random()
    rng.seed(time.time() + os.getpid())

    fail_count = 0

    for _ in range(NUM_SEQUENCES_PER_DESIGN):
        seq = generate_safe_sequence(ACTION_SPACE, SEQ_LENGTH)
        nodes, levels = run_abc_sequence(aig_file, seq, debug=False)

        if nodes is not None:
            final_product = nodes * levels
            initial_product = initial_nodes * initial_levels

            if initial_product > 0:
                score = final_product / initial_product
            else:
                score = 0

            candidates.append({
                "design": design_name,
                "initial_metrics": [initial_nodes, initial_levels],
                "final_metrics": [nodes, levels],
                "sequence": seq,
                "score": score
            })
        else:
            fail_count += 1

    candidates.sort(key=lambda x: x["score"])
    top_k = candidates[:KEEP_TOP_K]

    success_rate = 100 * (NUM_SEQUENCES_PER_DESIGN - fail_count) / NUM_SEQUENCES_PER_DESIGN

    if top_k:
        best = top_k[0]
        impr = 100 * (initial_nodes - best["final_metrics"][0]) / initial_nodes
        print(f"✅ {design_name}: 成功率 {success_rate:.0f}% ({len(candidates)}/{NUM_SEQUENCES_PER_DESIGN}). 最佳优化(Nodes) -{impr:.2f}%")
    else:
        print(f"⚠️ {design_name}: 0 条有效序列! (失败率 100%) - 可能是某个特定指令导致该电路 Crash")

    return top_k

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 找不到输入目录 {INPUT_DIR}")
        return

    if not os.path.exists(os.path.dirname(OUTPUT_DATA_FILE)):
        os.makedirs(os.path.dirname(OUTPUT_DATA_FILE))

    aig_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(".aig")]
    print(f"🔍 扫描到 {len(aig_files)} 个 AIG 文件")
    print("--------------------------------------------------")

    total_saved = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_design, aig): aig for aig in aig_files}

        with open(OUTPUT_DATA_FILE, 'a', encoding='utf-8') as f_out:
            for future in as_completed(futures):
                try:
                    result_top_k = future.result()
                    if result_top_k:
                        for item in result_top_k:
                            f_out.write(json.dumps(item) + "\n")
                        f_out.flush()
                        total_saved += len(result_top_k)
                except Exception as e:
                    print(f"❌ 线程严重错误: {e}")

    print("--------------------------------------------------")
    print(f"🎉 结束。共保存 {total_saved} 条数据。")

if __name__ == "__main__":
    main()
