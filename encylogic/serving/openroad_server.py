import os
import re
import uuid
import asyncio
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from encylogic import config

app = FastAPI(title="OpenROAD EDA Service (ASAP7) - Async Edition")

class EvaluateRequest(BaseModel):
    design: str
    sequence: List[str]

@app.post("/evaluate")
async def evaluate_sequence(req: EvaluateRequest):
    design = req.design
    seq_str = " ; ".join(req.sequence)

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    FLOW_DIR = str(config.ORFS_FLOW)

    print(f"📥 [收到任务] {design} (ASAP7) | RunID: {run_id}")

    original_v = f"{config.DESIGNS_DIR}/{design}/{design}.v"
    optimized_v = f"{config.DESIGNS_DIR}/{design}/{design}_opt_{run_id}.v"

    safe_design_name = f"{design}_top_module"

    default_error_response = {
        "status": "error", "wns": -9999.0, "tns": -9999.0, "delay": -9999.0, "area": -9999.0,
        "total_power": -9999.0, "leakage_power": -9999.0, "wirelength": -9999.0, "drc": -9999.0
    }

    default_timeout_response = {
        "status": "timeout", "wns": -9999.0, "tns": -9999.0, "delay": -9999.0, "area": -9999.0,
        "total_power": -9999.0, "leakage_power": -9999.0, "wirelength": -9999.0, "drc": -9999.0
    }

    try:
        print(f"⚙️ [阶段 1] {run_id} 正在独立运行自定义 ABC 逻辑优化...")

        ABC_PATH = config.YOSYS_ABC_BIN
        abc_cmd = f'{ABC_PATH} -c "source {config.DESIGNS_DIR}/abc.rc; read {original_v}; strash; &get; {seq_str}; &put; write_verilog {optimized_v}"'

        try:
            proc_abc = await asyncio.create_subprocess_shell(
                abc_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc_abc.communicate(), timeout=3000.0)

            if not os.path.exists(optimized_v):
                print(f"❌ [ABC 优化失败] {run_id} 未生成优化后的 Verilog！")
                return default_error_response

            with open(optimized_v, 'r', encoding='utf-8') as f:
                v_content = f.read()
            v_content = re.sub(rf'\bmodule\s+{re.escape(design)}\b', f'module {safe_design_name}', v_content)
            with open(optimized_v, 'w', encoding='utf-8') as f:
                f.write(v_content)

            print(f"✅ [阶段 1] {run_id} 逻辑优化完成，临时文件已净化为: {safe_design_name}")

        except asyncio.TimeoutError:
            print(f"⏰ [ABC 超时] {run_id} 逻辑优化陷入死循环！强行击杀进程...")
            try:
                proc_abc.kill()
            except ProcessLookupError:
                pass
            return default_timeout_response
        except Exception as e:
            print(f"❌ [ABC 运行报错] {run_id}: {str(e)}")
            return default_error_response

        print(f"🚀 [阶段 2] {run_id} 启动 OpenROAD 物理综合...")
        env = os.environ.copy()

        if "CUSTOM_ABC_SCRIPT" in env:
            del env["CUSTOM_ABC_SCRIPT"]

        env["NUM_CORES"] = "1"
        env["MAX_CORES"] = "1"
        env["OMP_NUM_THREADS"] = "1"

        cmd = f"make DESIGN_CONFIG={config.DESIGNS_DIR}/{design}/config.mk PLATFORM={config.PLATFORM} DESIGN_NAME={safe_design_name} VERILOG_FILES={optimized_v} FLOW_VARIANT={run_id}"

        try:
            proc_or = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=FLOW_DIR
            )

            stdout_or_bytes, stderr_or_bytes = await asyncio.wait_for(proc_or.communicate(), timeout=3600.0)
            stdout_str = stdout_or_bytes.decode('utf-8', errors='replace')
            stderr_str = stderr_or_bytes.decode('utf-8', errors='replace')

            log_candidates = [
                f"{FLOW_DIR}/reports/{config.PLATFORM}/{safe_design_name}/{run_id}/6_1_merged_metrics.json",
                f"{FLOW_DIR}/reports/{config.PLATFORM}/{safe_design_name}/{run_id}/6_finish.json",
                f"{FLOW_DIR}/reports/{config.PLATFORM}/{safe_design_name}/{run_id}/6_finish.rpt",
                f"{FLOW_DIR}/reports/{config.PLATFORM}/{safe_design_name}/{run_id}/6_final_report.log",
                f"{FLOW_DIR}/logs/{config.PLATFORM}/{safe_design_name}/{run_id}/5_2_route.log",
                f"{FLOW_DIR}/logs/{config.PLATFORM}/{safe_design_name}/{run_id}/6_finish.log",
                f"{FLOW_DIR}/reports/{config.PLATFORM}/{safe_design_name}/{run_id}/2_floorplan_final.rpt"
            ]

            wns_val, tns_val, delay_val, area_val = None, None, None, None
            total_power_val, leakage_power_val, wirelength_val, drc_val = None, None, None, None
            found_log_path = None

            for path in log_candidates:
                if not os.path.exists(path):
                    continue

                found_log_path = path

                if path.endswith('.json'):
                    with open(path, 'r', encoding='utf-8') as f:
                        try:
                            metrics = json.load(f)
                            if wns_val is None: wns_val = metrics.get('finish__timing__setup__ws')
                            if tns_val is None: tns_val = metrics.get('finish__timing__setup__tns')
                            if area_val is None: area_val = metrics.get('finish__design__core__area')
                            if total_power_val is None: total_power_val = metrics.get('finish__power__total')
                            if leakage_power_val is None: leakage_power_val = metrics.get('finish__power__leakage__total')
                            if drc_val is None:
                                drc_val = (metrics.get('finish__timing__drv__max_slew', 0) +
                                           metrics.get('finish__timing__drv__max_cap', 0) +
                                           metrics.get('finish__timing__drv__max_fanout', 0))
                            if delay_val is None and metrics.get('finish__timing__fmax'):
                                delay_val = 1.0 / metrics.get('finish__timing__fmax') * 1e12
                        except json.JSONDecodeError:
                            pass
                    continue

                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()

                    if wns_val is None:
                        wns_match = re.search(r'(?i)wns\s*max\s+([+-]?\d+(?:\.\d+)?)', content)
                        if wns_match: wns_val = float(wns_match.group(1))

                    if tns_val is None:
                        tns_match = re.search(r'(?i)tns\s*max\s+([+-]?\d+(?:\.\d+)?)', content)
                        if tns_match: tns_val = float(tns_match.group(1))

                    if delay_val is None:
                        delay_match = re.search(r'(?i)critical\s+path\s+delay\s*\n-+\n\s*([+-]?\d+(?:\.\d+)?)', content)
                        if delay_match: delay_val = float(delay_match.group(1))

                    if area_val is None:
                        area_match = re.findall(r'(?i)Design\s+area\s+([+-]?\d+(?:\.\d+)?)\s+um\^2', content)
                        if area_match: area_val = float(area_match[-1])

                    if total_power_val is None or leakage_power_val is None:
                        power_match = re.findall(r'(?im)^Total\s+[\d.eE+-]+\s+[\d.eE+-]+\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+[\d.]+\s*%', content)
                        if power_match:
                            leakage_power_val = float(power_match[-1][0])
                            total_power_val = float(power_match[-1][1])

                    if wirelength_val is None:
                        wl_match = re.findall(r'(?i)Total\s+wire\s*length\s*=\s*([+-]?\d+(?:\.\d+)?)\s*um', content)
                        if wl_match: wirelength_val = float(wl_match[-1])

                    if drc_val is None:
                        drc_match = re.findall(r'(?i)Number\s+of\s+violations\s*=\s*(\d+)', content)
                        if drc_match: drc_val = int(drc_match[-1])

                if all(v is not None for v in [wns_val, tns_val, delay_val, area_val, total_power_val, wirelength_val, drc_val]):
                    break

            def get_val(v): return v if v is not None else -9999.0

            wns_val = get_val(wns_val)
            tns_val = get_val(tns_val)
            delay_val = get_val(delay_val)
            area_val = get_val(area_val)
            total_power_val = get_val(total_power_val)
            leakage_power_val = get_val(leakage_power_val)
            wirelength_val = get_val(wirelength_val)
            drc_val = get_val(drc_val)

            if found_log_path and (wns_val != -9999.0 or delay_val != -9999.0 or area_val != -9999.0):
                print(f"🎉 [完成] {design} ({run_id}) | WNS: {wns_val} | TNS: {tns_val} | Delay: {delay_val} | Area: {area_val}")
            else:
                print(f"⚠️ [警告] {run_id} 布线完成，但在日志中未能完全解析出时序或面积数据。")
                print(f"🔴 [OpenROAD STDERR 报错信息]:\n{stderr_str[-1500:]}")
                print(f"🟡 [OpenROAD STDOUT 常规输出]:\n{stdout_str[-1500:]}")

            return {
                "status": "success",
                "wns": wns_val,
                "tns": tns_val,
                "delay": delay_val,
                "area": area_val,
                "total_power": total_power_val,
                "leakage_power": leakage_power_val,
                "wirelength": wirelength_val,
                "drc": drc_val
            }

        except asyncio.TimeoutError:
            print(f"⏰ [超时] {design} ({run_id}) 后端评估超时 3600s！强行击杀进程...")
            try:
                proc_or.kill()
            except ProcessLookupError:
                pass
            return default_timeout_response

        except Exception as e:
            print(f"❌ [OpenROAD 致命报错] {run_id}: {str(e)}")
            return default_error_response

    finally:
        if os.path.exists(optimized_v):
            try:
                os.remove(optimized_v)
            except Exception:
                pass

        cleanup_cmd = f"rm -rf {FLOW_DIR}/reports/{config.PLATFORM}/{safe_design_name}/{run_id} {FLOW_DIR}/logs/{config.PLATFORM}/{safe_design_name}/{run_id} {FLOW_DIR}/results/{config.PLATFORM}/{safe_design_name}/{run_id}"
        try:
            asyncio.create_task(
                asyncio.create_subprocess_shell(
                    cleanup_cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
            )
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT)
