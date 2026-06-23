import json
from encylogic import config

input_file = str(config.DATA_DIR / "top50_best_sequences.jsonl")
output_file = str(config.DATA_DIR / "top50_best_sequences_final.json")

data = []
print("正在转换格式...")
with open(input_file, 'r') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

with open(output_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"转换完成！标准 JSON 已保存至 {output_file}")
