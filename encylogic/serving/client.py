import json
import requests
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from encylogic import config

INPUT_JSONL = str(config.DATA_DIR / "angledata.jsonl")
OUTPUT_JSONL = str(config.DATA_DIR / "out_evaluated_angledata.jsonl")
API_URL = config.API_URL
MAX_WORKERS = 16

def send_evaluation_request(item):
    payload = {
        "design": item["design"],
        "sequence": item["sequence"]
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=2000)
        result = response.json()

        if result.get("status") == "success":
            item["finish_wns"] = result.get("wns")
            item["finish_tns"] = result.get("tns")
            item["finish_delay"] = result.get("delay")
            print(f"✅ {item['design']} 成功 | WNS: {item['finish_wns']} | Delay: {item['finish_delay']}ns")
        else:
            print(f"⚠️ {item['design']} 异常: {result}")
            item["finish_wns"], item["finish_tns"], item["finish_delay"] = -9999.0, -9999.0, -9999.0

    except Exception as e:
        print(f"❌ 请求失败: {e}")
        item["finish_wns"], item["finish_tns"], item["finish_delay"] = -9999.0, -9999.0, -9999.0

    return item

def main():
    print("📦 正在扫描已处理的数据 (断点续传检查)...")
    processed_keys = set()

    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    key = data["design"] + "".join(data["sequence"])
                    processed_keys.add(key)
    print(f"🔍 发现已完成 {len(processed_keys)} 条记录。")

    print("📦 正在加载全量测试数据...")
    test_data = []

    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                key = item["design"] + "".join(item["sequence"])
                if key not in processed_keys:
                    test_data.append(item)

    if not test_data:
        print("🎉 所有数据都已经评估完毕，无需重复运行！")
        return

    print(f"🚀 开始向 OpenROAD 发送 {len(test_data)} 个并发请求 (128核并行)...")
    start_time = time.time()

    with open(OUTPUT_JSONL, 'a', encoding='utf-8') as out_f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(send_evaluation_request, item) for item in test_data]

            for future in as_completed(futures):
                res = future.result()

                out_f.write(json.dumps(res) + "\n")
                out_f.flush()

    print(f"🎉 全部测试完毕！本次耗时: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
