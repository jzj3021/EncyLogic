import json
from collections import defaultdict
from encylogic import config

INPUT_JSONL = str(config.DATA_DIR / "out_evaluated_angledata.jsonl")

OUTPUT_JSONL = str(config.DATA_DIR / "top50_best_sequences.jsonl")

def process_data():
    print("📦 正在加载评估完成的序列数据...")
    design_groups = defaultdict(list)

    try:
        with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                design = data.get("design", "unknown")
                design_groups[design].append(data)
    except FileNotFoundError:
        print(f"❌ 找不到输入文件: {INPUT_JSONL}，请确保路径正确！")
        return

    print(f"📊 共读取到 {len(design_groups)} 种不同的电路。")

    top_half_results = []

    for design, items in design_groups.items():
        sorted_items = sorted(
            items,
            key=lambda x: x.get("finish_delay", -9999.0) if x.get("finish_delay", -9999.0) > 0 else float('inf')
        )

        half_idx = len(sorted_items) // 2
        best_half = sorted_items[:half_idx]

        valid_count = 0
        for item in best_half:
            delay = item.get("finish_delay", -9999.0)

            if delay > 0:
                original_score = item.get("score", 1.0)
                item["score"] = original_score * delay
                valid_count += 1

            top_half_results.append(item)

        print(f"✅ [{design}] 淘汰完毕：原 {len(items)} 条 -> 晋级 {len(best_half)} 条 (有效 Delay: {valid_count}条)")

    print(f"\n💾 正在将提炼后的黄金序列写入磁盘...")
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for res in top_half_results:
            f.write(json.dumps(res) + "\n")

    print(f"🎉 大功告成！所有电路的 Top 50% 最佳序列已保存至: {OUTPUT_JSONL}")
    print(f"🏆 总计保留了 {len(top_half_results)} 条高质量数据供模型训练！")

if __name__ == "__main__":
    process_data()
