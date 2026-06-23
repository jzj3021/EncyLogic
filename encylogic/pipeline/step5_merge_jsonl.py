import json
import os
from encylogic import config

input_files = [
    str(config.DATA_DIR / "angel_encyclopedia_train.jsonl"),
    str(config.DATA_DIR / "angel_encyclopedia.jsonl"),
]

output_file = str(config.DATA_DIR / "merged_valid.jsonl")

error_file = str(config.DATA_DIR / "merged_errors.jsonl")

def merge_and_validate():
    success_count = 0
    error_count = 0

    open(output_file, 'w', encoding='utf-8').close()
    open(error_file, 'w', encoding='utf-8').close()

    print(f"开始合并 {len(input_files)} 个文件...")

    with open(output_file, 'a', encoding='utf-8') as f_out, \
         open(error_file, 'a', encoding='utf-8') as f_err:

        for file_path in input_files:
            print(f"正在处理: {file_path}")

            if not os.path.exists(file_path):
                print(f"⚠️ 警告: 文件不存在，跳过 - {file_path}")
                continue

            with open(file_path, 'r', encoding='utf-8') as f_in:
                for line_num, line in enumerate(f_in, 1):
                    stripped_line = line.strip()

                    if not stripped_line:
                        continue

                    try:
                        json.loads(stripped_line)

                        f_out.write(stripped_line + '\n')
                        success_count += 1

                    except json.JSONDecodeError as e:
                        error_msg = {
                            "source_file": file_path,
                            "line_number": line_num,
                            "error": str(e),
                            "raw_content": stripped_line
                        }
                        f_err.write(json.dumps(error_msg, ensure_ascii=False) + '\n')
                        error_count += 1

    print("-" * 30)
    print("合并完成！")
    print(f"✅ 成功合并行数: {success_count} (保存至 {output_file})")
    if error_count > 0:
        print(f"❌ 发现错误行数: {error_count} (详情见 {error_file})")
    else:
        print("🎉 没有发现任何格式错误。")

if __name__ == "__main__":
    merge_and_validate()
