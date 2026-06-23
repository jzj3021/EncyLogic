import os
import glob
import subprocess
import re

from encylogic import config

ABC_CMD = config.ABC_BIN
RESULTS_DIR = str(config.RESULTS_DIR)
LIB_PATH = str(config.LIB_PATH)

def evaluate_aig(aig_path):
    and_nodes, aig_levels = -1, -1
    luts, fpga_levels = -1, -1
    area, delay = -1.0, -1.0

    if not os.path.exists(aig_path):
        print(f"🔴 [AIG 文件读取失败] 找不到文件: {aig_path}")
        return and_nodes, aig_levels, luts, fpga_levels, area, delay

    cmd_aig = f"read {aig_path}; print_stats"
    res_aig = subprocess.run([ABC_CMD, "-c", cmd_aig], capture_output=True, text=True)

    match_aig = re.search(r'nd\s*=\s*(\d+).*?lev\s*=\s*(\d+)', res_aig.stdout)
    if match_aig:
        and_nodes, aig_levels = int(match_aig.group(1)), int(match_aig.group(2))

    cmd_fpga = f"read {aig_path}; if -K 6; print_stats"
    res_fpga = subprocess.run([ABC_CMD, "-c", cmd_fpga], capture_output=True, text=True)

    match_fpga = re.search(r'nd\s*=\s*(\d+).*?lev\s*=\s*(\d+)', res_fpga.stdout)
    if match_fpga:
        luts, fpga_levels = int(match_fpga.group(1)), int(match_fpga.group(2))

    if not os.path.exists(LIB_PATH):
        print(f"\n🔴 [ASIC 库文件错误] 系统找不到指定的 .lib 文件或无读取权限:")
        print(f"   路径: {LIB_PATH}")
    else:
        cmd_asic = f"read_lib {LIB_PATH}; read {aig_path}; map; print_stats"
        res_asic = subprocess.run([ABC_CMD, "-c", cmd_asic], capture_output=True, text=True)

        match_area = re.search(r'(?i)area\s*=\s*([+-]?\d+(?:\.\d+)?)', res_asic.stdout)
        match_delay = re.search(r'(?i)delay\s*=\s*([+-]?\d+(?:\.\d+)?)', res_asic.stdout)

        if match_area and match_delay:
            area = float(match_area.group(1))
            delay = float(match_delay.group(1))
        else:
            print(f"\n🔴 [ASIC 映射异常] {aig_path}")
            print(f"👉 执行指令: {cmd_asic}")
            if res_asic.stderr.strip():
                print(f"🔴 STDERR:\n{res_asic.stderr.strip()}\n")
            if res_asic.stdout.strip():
                print(f"🟡 STDOUT:\n{res_asic.stdout.strip()}\n")

    return and_nodes, aig_levels, luts, fpga_levels, area, delay

def main():
    aig_files = glob.glob(os.path.join(RESULTS_DIR, "*.aig"))
    if not aig_files:
        print(f"❌ 在 {RESULTS_DIR} 下没有找到优化后的 .aig 文件。")
        return

    print(f"\n{'Design':<15} | {'AND Nodes':<10} | {'AIG Lev':<10} | {'Area (ASIC)':<12} | {'Delay (ASIC)':<12} | {'LUTs (FPGA)':<12} | {'Lev (FPGA)':<10}")
    print("-" * 100)

    for aig in sorted(aig_files):
        design_name = os.path.basename(aig).replace("_optimized_delay.aig", "")
        and_nodes, aig_levels, luts, fpga_levels, area, delay = evaluate_aig(aig)
        print(f"{design_name:<15} | {and_nodes:<10} | {aig_levels:<10} | {area:<12.2f} | {delay:<12.2f} | {luts:<12} | {fpga_levels:<10}")

if __name__ == "__main__":
    main()
