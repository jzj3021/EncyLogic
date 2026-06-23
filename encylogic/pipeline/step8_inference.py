import torch
import torch.nn as nn
import subprocess
import re
import os
import shutil
import glob
import random
import math
from encylogic import config

EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 8
SEQ_LEN = 25

try:
    from encylogic.actionspace import ACTION_SPACE1
except ImportError:
    print("❌ 错误: 找不到 actionspace.py 文件或 ACTION_SPACE1 变量。")
    exit(1)

stoi = {cmd: i for i, cmd in enumerate(ACTION_SPACE1)}
itos = {i: cmd for i, cmd in enumerate(ACTION_SPACE1)}
BOS_IDX = stoi.get("<bos>", 0)
PAD_IDX = stoi.get("<pad>", 0)

class SeqFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(ACTION_SPACE1), EMBED_DIM)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, EMBED_DIM))
        decoder_layer = nn.TransformerDecoderLayer(d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=512, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=NUM_LAYERS)
        self.fc_out = nn.Linear(EMBED_DIM, len(ACTION_SPACE1))

    def forward(self, x):
        seq_len = x.size(1)
        safe_len = min(seq_len, self.pos_encoder.shape[1])
        emb = self.embedding(x) + self.pos_encoder[:, :safe_len, :]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        output = self.transformer_decoder(emb, memory=emb, tgt_mask=mask, memory_mask=mask)
        return self.fc_out(output)

def get_abc_command():
    return config.ABC_BIN
ABC_CMD = get_abc_command()

def get_stats(aig_path):
    """
    [修改版] 获取电路节点数和层级
    返回: (nodes, levels, metric)
    metric = nodes * levels
    """
    cmd = f"read {aig_path}; strash; &get; &put; print_stats;"
    try:
        res = subprocess.run([ABC_CMD, "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        nd_match = re.search(r'nd\s*=\s*(\d+)', res.stdout)
        lev_match = re.search(r'lev\s*=\s*(\d+)', res.stdout)

        if nd_match and lev_match:
            nd = int(nd_match.group(1))
            lev = int(lev_match.group(1))
            metric = nd * lev
            return nd, lev, metric
    except:
        pass
    return 999999, 999999, float('inf')

def apply_action(input_aig, output_aig, action):
    """
    [GIA 适配版] 执行单个操作
    """
    cmd = f"read {input_aig}; strash; &get; {action}; &put; write_aiger {output_aig};"
    subprocess.run([ABC_CMD, "-c", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def optimize_circuit(target_circuit_path, model, device):
    design_name = os.path.basename(target_circuit_path)
    work_dir = "temp_inference"
    if not os.path.exists(work_dir): os.makedirs(work_dir)

    current_aig = os.path.join(work_dir, "current.aig")
    shutil.copy(target_circuit_path, current_aig)

    init_nd, init_lev, init_metric = get_stats(current_aig)

    current_nd, current_lev, current_metric = init_nd, init_lev, init_metric

    print(f"\n🎯 目标电路: {design_name}")
    print(f"   初始状态: Nodes={init_nd}, Level={init_lev} | ADP(N*L)={init_metric}")

    generated_actions = []

    model.eval()
    with torch.no_grad():
        input_seq = [BOS_IDX]

        for step in range(SEQ_LEN):
            tensor_in = torch.tensor([input_seq], dtype=torch.long).to(device)
            logits = model(tensor_in)

            next_token_logits = logits[0, -1, :]

            for token_key in ["<pad>", "<bos>", "<eos>"]:
                if token_key in stoi:
                    next_token_logits[stoi[token_key]] = -float('inf')

            topk_probs, topk_indices = torch.topk(next_token_logits, 3)

            best_step_action = None
            best_step_metric = current_metric
            best_step_aig = None

            print(f" Step {step+1} 尝试候选: ", end="")
            candidate_metrics = []

            for idx in topk_indices:
                action_name = itos[idx.item()]
                print(f"[{action_name}] ", end="")

                test_aig = os.path.join(work_dir, f"test_{step}_{action_name}.aig")
                apply_action(current_aig, test_aig, action_name)

                nd, lev, metric = get_stats(test_aig)
                candidate_metrics.append((action_name, metric))

                if metric < best_step_metric:
                    best_step_metric = metric
                    best_step_action = action_name
                    best_step_aig = test_aig
                    step_nd, step_lev = nd, lev

            print("")

            if best_step_action:
                print(f"   ✅ 采纳 '{best_step_action}'")
                print(f"      (N*L: {current_metric} -> {best_step_metric} | Nd:{step_nd}, Lev:{step_lev})")

                current_metric = best_step_metric
                current_nd = step_nd
                current_lev = step_lev

                generated_actions.append(best_step_action)
                input_seq.append(stoi[best_step_action])

                shutil.move(best_step_aig, current_aig)
            else:
                fallback_action = itos[topk_indices[0].item()]
                print(f"   ⚠️ 暂无优化 (Min Metric: {min([m[1] for m in candidate_metrics])}), 强制执行 Top1 '{fallback_action}'")

                apply_action(current_aig, current_aig, fallback_action)
                current_nd, current_lev, current_metric = get_stats(current_aig)

                generated_actions.append(fallback_action)
                input_seq.append(topk_indices[0].item())

    print("-" * 30)
    print(f"🎉 优化完成！")
    print(f"原始: Nd={init_nd}, Lev={init_lev} (ADP={init_metric})")
    print(f"最终: Nd={current_nd}, Lev={current_lev} (ADP={current_metric})")

    if init_metric > 0:
        improvement = (init_metric - current_metric) / init_metric * 100
        print(f"🚀 ADP优化率: {improvement:.2f}%")

        nd_imp = (init_nd - current_nd) / init_nd * 100
        print(f"📉 面积优化率: {nd_imp:.2f}%")

    print(f"策略序列: {' ; '.join(generated_actions)}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    model = SeqFormer().to(device)
    model_path = str(config.GRPO_CHECKPOINT)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("✅ 成功加载 RL 模型")
        except Exception as e:
            print(f"❌ 加载模型报错: {e}")
            exit()
    else:
        print(f"❌ 找不到模型文件 {model_path}")
        exit()

    test_target = str(config.BENCHMARK_DIR / "max.aig")
    if not os.path.exists(test_target):
        candidates = glob.glob("data/b_Rn/data1aig/*.aig") + glob.glob("*.aig")
        if candidates:
            test_target = random.choice(candidates)
        else:
            print("❌ 没有找到可用的 .aig 测试文件")
            exit()

    optimize_circuit(test_target, model, device)
