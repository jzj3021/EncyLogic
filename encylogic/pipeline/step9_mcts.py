import torch
import torch.nn as nn
import subprocess
import re
import os
import shutil
import math
import copy
import numpy as np

from encylogic import config

EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 8
SEQ_LEN = 25
MCTS_DEPTH = 50
MCTS_BREADTH = 20
MCTS_ITERATIONS = 50
UCT_C = 1.4

try:
    from encylogic.actionspace import ACTION_SPACE1
except ImportError:
    print("❌ 错误: 找不到 actionspace.py 文件。")
    if 'ACTION_SPACE1' not in locals():
        ACTION_SPACE1 = ["&balance", "&rewrite", "&refactor", "&resub", "&mfs"]
        print("⚠️ 使用临时 GIA Action Space 进行演示")

stoi = {cmd: i for i, cmd in enumerate(ACTION_SPACE1)}
itos = {i: cmd for i, cmd in enumerate(ACTION_SPACE1)}
BOS_IDX = stoi.get("<bos>", 0)
EOS_IDX = stoi.get("<eos>", 0)
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
        emb = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        output = self.transformer_decoder(emb, memory=emb, tgt_mask=mask, memory_mask=mask)
        return self.fc_out(output)

def get_abc_command():
    if os.path.exists(config.ABC_BIN): return config.ABC_BIN
    return config.ABC_BIN
ABC_CMD = get_abc_command()

def get_stats(aig_path):
    """
    [修改点 2] 获取 ADP (Area-Delay Product) = Node * Level
    """
    cmd = f"read {aig_path}; strash; print_stats;"
    try:
        res = subprocess.run([ABC_CMD, "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        nd_match = re.search(r'nd\s*=\s*(\d+)', res.stdout)
        lev_match = re.search(r'lev\s*=\s*(\d+)', res.stdout)

        if nd_match and lev_match:
            nd = int(nd_match.group(1))
            lev = int(lev_match.group(1))
            cost = nd * lev
            return cost
    except Exception as e:
        print(f"Error parsing stats: {e}")
        pass
    return 99999999

def apply_sequence(input_aig, output_aig, action_indices):
    """
    [修改点 3] 在 GIA 域执行操作序列
    """
    cmds = []
    for idx in action_indices:
        if idx in itos and itos[idx] not in ["<pad>", "<bos>", "<eos>"]:
            cmds.append(itos[idx])

    if not cmds:
        shutil.copy(input_aig, output_aig)
        return

    ops_str = " ; ".join(cmds)

    full_cmd = f"read {input_aig}; &get; {ops_str}; &put; write_aiger {output_aig};"

    subprocess.run([ABC_CMD, "-c", full_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

class MCTSNode:
    def __init__(self, seq_indices, parent=None):
        self.seq_indices = seq_indices
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.probs = None

    def is_leaf(self):
        return len(self.children) == 0

class SeqTree:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def search(self, start_seq_indices, depth=MCTS_DEPTH, breadth=MCTS_BREADTH):
        root = MCTSNode(start_seq_indices)
        candidates = []

        for i in range(breadth):
            node = root
            for d in range(depth):
                if node.is_leaf():
                    self._expand(node)

                action_idx, next_node = self._select(node)
                node = next_node

                if itos.get(action_idx) in ["<eos>", "<pad>"]:
                    break

            candidates.append(node.seq_indices)
            self._backpropagate(node)

        return candidates

    def _expand(self, node):
        self.model.eval()
        with torch.no_grad():
            inp = torch.tensor([node.seq_indices], dtype=torch.long).to(self.device)
            logits = self.model(inp)
            next_token_logits = logits[0, -1, :]

            for token in ["<pad>", "<bos>", "<eos>"]:
                if token in stoi:
                    next_token_logits[stoi[token]] = -float('inf')

            probs = torch.softmax(next_token_logits, dim=0).cpu().numpy()
            node.probs = probs

    def _select(self, node):
        best_score = -float('inf')
        best_action = None
        top_indices = np.argsort(node.probs)[-10:]

        for action_idx in top_indices:
            if action_idx not in node.children:
                new_seq = node.seq_indices + [action_idx]
                node.children[action_idx] = MCTSNode(new_seq, parent=node)

            child = node.children[action_idx]
            prior = node.probs[action_idx]
            uct = UCT_C * prior * math.sqrt(node.visits) / (1 + child.visits)

            if uct > best_score:
                best_score = uct
                best_action = action_idx

        return best_action, node.children[best_action]

    def _backpropagate(self, node):
        curr = node
        while curr is not None:
            curr.visits += 1
            curr = curr.parent

import torch
import torch.nn as nn
import subprocess
import re
import os
import shutil
import math
import copy
import numpy as np
import glob
import random

def optimize_with_mcts(target_aig, model, device):
    design_name = os.path.basename(target_aig)
    print(f"\n 启动 SeqTree MCTS 优化 (GIA Mode): {design_name}")

    work_dir = "temp_mcts"
    if not os.path.exists(work_dir): os.makedirs(work_dir)

    current_aig = os.path.join(work_dir, "current.aig")
    shutil.copy(target_aig, current_aig)

    initial_score = get_stats(current_aig)
    best_score = initial_score
    print(f"   初始 ADP (N*L): {initial_score}")

    current_seq = [BOS_IDX]
    mcts = SeqTree(model, device)

    for i in range(MCTS_ITERATIONS):
        print(f"\n[Iteration {i+1}/{MCTS_ITERATIONS}] 正在思考...")

        candidate_seqs = mcts.search(current_seq)

        best_cand_idx = -1
        best_cand_score = float('inf')

        for idx, seq in enumerate(candidate_seqs):
            test_aig = os.path.join(work_dir, f"test_cand_{idx}.aig")
            new_actions = seq[len(current_seq):]
            if not new_actions: continue

            apply_sequence(current_aig, test_aig, new_actions)
            score = get_stats(test_aig)

            if score < best_cand_score:
                best_cand_score = score
                best_cand_idx = idx

        if best_cand_idx != -1 and best_cand_score < best_score:
            best_seq = candidate_seqs[best_cand_idx]
            new_ops_indices = best_seq[len(current_seq):]
            new_ops_names = [itos[ix] for ix in new_ops_indices]

            print(f"   ✅ 采纳最佳方案 (ADP: {best_score} -> {best_cand_score})")
            print(f"      新增操作: {' ; '.join(new_ops_names)}")

            best_score = best_cand_score
            current_seq = best_seq

            best_cand_aig = os.path.join(work_dir, f"test_cand_{best_cand_idx}.aig")
            shutil.move(best_cand_aig, current_aig)
        else:
            print("   ⚠️ 本轮未找到更优方案，保持现状。")

    print("-" * 30)
    improvement = (initial_score - best_score) / initial_score * 100
    print(f"🚀 最终优化率: {improvement:.2f}% ({initial_score} -> {best_score})")

    results_dir = str(config.RESULTS_DIR)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    base_name = os.path.basename(target_aig)
    name_root, _ = os.path.splitext(base_name)
    output_filename = f"{name_root}_optimized.aig"
    final_output_path = os.path.join(results_dir, output_filename)

    try:
        shutil.copy(current_aig, final_output_path)
        print(f"\n💾 结果已成功保存至: {final_output_path}")

        log_path = os.path.join(results_dir, "optimization_log.txt")
        with open(log_path, "a") as f:
            f.write(f"{base_name}: Initial={initial_score}, Final={best_score}, Imp={improvement:.2f}%\n")

    except Exception as e:
        print(f"❌ 保存文件失败: {e}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SeqFormer().to(device)
    if os.path.exists(str(config.GRPO_CHECKPOINT)):
        model.load_state_dict(torch.load(str(config.GRPO_CHECKPOINT), map_location=device))

    if not os.path.exists(str(config.BENCHMARK_DIR)):
        os.makedirs(str(config.BENCHMARK_DIR), exist_ok=True)
        print("请在 benchmark 目录放入 .aig 文件")

    aigs = glob.glob(str(config.BENCHMARK_DIR / "adder.aig"))
    if aigs:
        target = random.choice(aigs)
        optimize_with_mcts(target, model, device)
    else:
        print("❌ 找不到测试电路")
