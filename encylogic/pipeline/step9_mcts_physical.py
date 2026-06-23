import torch
import torch.nn as nn
import subprocess
import os
import shutil
import math
import numpy as np
import glob
import random
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor

from encylogic import config

EMBED_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 6
SEQ_LEN = 25

SEARCH_BREADTH = 16
SEARCH_DEPTH = 25
SEARCH_ITERATIONS = 10
UCT_C = 1.4

MAX_WORKERS = 16

session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS))

try:
    from encylogic.actionspace import ACTION_SPACE1
except ImportError:
    print("❌ 错误: 找不到 actionspace.py 文件。")
    exit(1)

stoi = {cmd: i for i, cmd in enumerate(ACTION_SPACE1)}
itos = {i: cmd for i, cmd in enumerate(ACTION_SPACE1)}
BOS_IDX = stoi.get("<bos>", 0)
EOS_IDX = stoi.get("<eos>", 0)
PAD_IDX = stoi.get("<pad>", 0)
VOCAB_SIZE = len(ACTION_SPACE1)

class SeqFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_emb = nn.Embedding(2048, EMBED_DIM)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=EMBED_DIM * 4,
            batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.fc_out = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x_emb = self.tok_emb(x) + self.pos_emb(positions)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        key_padding_mask = (x == PAD_IDX)

        out = self.transformer(x_emb, mask=causal_mask, src_key_padding_mask=key_padding_mask, is_causal=True)
        return self.fc_out(out)

def api_eval_worker(args):
    """独立的物理评估 Worker，返回 (idx, delay, area)"""
    idx, design_name, cmds = args

    if not cmds:
        cmds = ["print_stats"]

    payload = {"design": design_name, "sequence": cmds}
    try:
        response = session.post(config.API_URL, json=payload, timeout=(10, 3600))
        result = response.json()

        if result.get("status") == "success":
            delay = float(result.get("delay", 999999.0))
            area = float(result.get("area", 999999.0))

            if delay == -9999.0:
                print(f"❌ [评估失败] 后端时序解析失败 (返回 -9999.0)")
                return idx, float('inf'), float('inf')

            return idx, delay, area
        else:
            print(f"❌ [评估失败] Docker 返回异常状态: {result}")
            return idx, float('inf'), float('inf')

    except Exception as e:
        print(f"❌ [API 致命错误] 无法连接到 Docker (网络断开或超时): {e}")
        return idx, float('inf'), float('inf')

ABC_CMD = config.ABC_BIN
def apply_sequence_local(input_aig, output_aig, cmds):
    """仅用于最终本地保存 AIG"""
    if not cmds:
        shutil.copy(input_aig, output_aig)
        return
    ops_str = " ; ".join(cmds)
    full_cmd = f"read {input_aig}; &get; {ops_str}; &put; write {output_aig};"
    subprocess.run([ABC_CMD, "-c", full_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

class SearchNode:
    def __init__(self, seq_indices, parent=None, action_idx=None):
        self.seq_indices = seq_indices
        self.parent = parent
        self.action_idx = action_idx
        self.children = {}
        self.visits = 0
        self.probs = None
        self.q_value = 0.0
        self.total_value = 0.0
        self.virtual_loss = 0

    def is_leaf(self):
        return len(self.children) == 0

class PolicyGuidedSearchTree:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.root = None

    def get_batched_candidates(self, root_seq, batch_size, depth):
        """批量获取候选叶子节点，并加上虚拟惩罚"""
        if self.root is None or self.root.seq_indices != root_seq:
            self.root = SearchNode(root_seq)

        leaves = []
        for _ in range(batch_size):
            node = self.root
            for _ in range(depth):
                if node.is_leaf():
                    self._expand(node)

                action_idx, next_node = self._select(node)
                next_node.virtual_loss += 1
                node = next_node

                if itos.get(action_idx) in ["<eos>", "<pad>"]:
                    break
            leaves.append(node)
        return leaves

    def _expand(self, node):
        self.model.eval()
        with torch.no_grad():
            inp = torch.tensor([node.seq_indices], dtype=torch.long).to(self.device)
            logits = self.model(inp)
            next_token_logits = logits[0, -1, :]
            for token in ["<pad>", "<bos>"]:
                if token in stoi:
                    next_token_logits[stoi[token]] = -float('inf')
            temperature = 1.5
            probs = torch.softmax(next_token_logits / temperature, dim=0).cpu().numpy()
            if node.parent is None:
                noise = np.random.dirichlet([0.3] * len(probs))
                probs = 0.75 * probs + 0.25 * noise
            node.probs = probs

    def _select(self, node):
        best_score = -float('inf')
        best_action = None
        top_indices = np.argsort(node.probs)[-10:]

        for action_idx in top_indices:
            if action_idx not in node.children:
                new_seq = node.seq_indices + [action_idx]
                node.children[action_idx] = SearchNode(new_seq, parent=node, action_idx=action_idx)

            child = node.children[action_idx]
            prior = node.probs[action_idx]

            effective_visits = child.visits + child.virtual_loss
            uct = child.q_value + UCT_C * prior * math.sqrt(max(1, node.visits)) / (1 + effective_visits)

            if uct > best_score:
                best_score = uct
                best_action = action_idx

        return best_action, node.children[best_action]

    def backpropagate_batch(self, leaves, rewards):
        """将 Docker 返回的物理 Reward 反向传播回树中"""
        for leaf, reward in zip(leaves, rewards):
            curr = leaf
            while curr is not None:
                if curr.virtual_loss > 0:
                    curr.virtual_loss -= 1
                curr.visits += 1
                curr.total_value += reward
                curr.q_value = curr.total_value / curr.visits
                curr = curr.parent

def optimize_with_guided_search(target_aig, model, device):
    global_start_time = time.time()

    design_name = os.path.basename(target_aig)
    name_root, _ = os.path.splitext(design_name)

    print(f"\n🚀 启动策略指导采样搜索 (目标: 真实物理 Delay) | 目标设计: {name_root}")

    work_dir = "temp_search"
    os.makedirs(work_dir, exist_ok=True)
    current_aig = os.path.join(work_dir, "current.aig")
    shutil.copy(target_aig, current_aig)

    print("⏳ 正在获取初始物理 Baseline...")
    _, initial_score, initial_area = api_eval_worker((0, name_root, []))

    if initial_score == float('inf'):
        print("❌ 无法获取初始物理延迟，Docker API 可能出错了！")
        return

    best_score = initial_score
    print(f"📉 初始物理 Baseline -> Delay: {initial_score} ps | Area: {initial_area}")

    current_seq = [BOS_IDX]
    search_tree = PolicyGuidedSearchTree(model, device)

    MCTS_ROUNDS = 4

    no_improvement_count = 0

    for i in range(SEARCH_ITERATIONS):
        iter_start_time = time.time()
        print(f"\n[Iteration {i+1}/{SEARCH_ITERATIONS}] MCTS 树生长与物理评估交互中...")

        best_cand_idx = -1
        best_cand_score = float('inf')
        best_cand_area = float('inf')
        best_seq_this_iter = None

        for m_round in range(MCTS_ROUNDS):
            print(f"  🌳 [MCTS Round {m_round+1}/{MCTS_ROUNDS}] 采样并评估 {MAX_WORKERS} 条分支...")
            leaves = search_tree.get_batched_candidates(current_seq, MAX_WORKERS, SEARCH_DEPTH)

            tasks = []
            for idx, leaf in enumerate(leaves):
                full_cmds = [itos.get(int(ix)) for ix in leaf.seq_indices if itos.get(int(ix), "") not in ["<pad>", "<eos>", "<bos>"]]
                tasks.append((idx, name_root, full_cmds))

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                results = list(executor.map(api_eval_worker, tasks))

            rewards = []
            for idx, delay, area in results:
                if delay < best_cand_score:
                    best_cand_score = delay
                    best_cand_idx = idx
                    best_cand_area = area
                    best_seq_this_iter = leaves[idx].seq_indices

                if delay == float('inf'):
                    reward = -1.0
                else:
                    reward = (initial_score - delay) / initial_score * 10.0
                    reward = max(-1.0, reward)

                rewards.append(reward)

            search_tree.backpropagate_batch(leaves, rewards)

        if best_seq_this_iter is not None and best_cand_score < best_score:
            new_ops_names = [itos.get(int(ix)) for ix in best_seq_this_iter[len(current_seq):] if itos.get(int(ix), "") not in ["<pad>", "<eos>", "<bos>"]]

            print(f"  ✅ 物理环境验证突破！")
            print(f"    📈 Delay: {best_score} ps -> {best_cand_score} ps | 当前 Area: {best_cand_area}")
            print(f"  💡 新增采纳操作: {' ; '.join(new_ops_names)}")

            best_score = best_cand_score
            current_seq = best_seq_this_iter
            apply_sequence_local(current_aig, current_aig, new_ops_names)

            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= 8:
                print(f"  ⚠️ 经过连续 {no_improvement_count} 轮评估均未超越当前最优 Delay，搜索趋于收敛，提前结束。")
                break
            else:
                print(f"  ⚠️ 第 {no_improvement_count} 轮未超越当前最优 Delay，将再尝试一次搜索...")

        print(f"  ⏱️ 本轮耗时: {time.time() - iter_start_time:.2f}s")

    total_time = time.time() - global_start_time
    improvement = (initial_score - best_score) / initial_score * 100 if initial_score > 0 else 0

    print("-" * 40)
    print(f"🏆 搜索完成: {name_root}")
    print(f"📈 最终 Delay 优化率: {improvement:.2f}% ({initial_score}ps -> {best_score}ps)")
    print(f"⏱️ 总耗时: {total_time:.2f} 秒")
    print("-" * 40)

    results_dir = str(config.RESULTS_DIR)
    os.makedirs(results_dir, exist_ok=True)
    shutil.copy(current_aig, os.path.join(results_dir, f"{name_root}_optimized_delay.aig"))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 运行推理设备: {device}")

    model = SeqFormer().to(device)
    model_path = str(config.GRPO_CHECKPOINT)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)

        old_pos_weight = state_dict.get('pos_emb.weight')
        if old_pos_weight is not None and old_pos_weight.shape[0] < 2048:
            print(f"⚠️ 检测到旧版位置编码尺寸 {old_pos_weight.shape[0]}，正在无损扩容至 2048...")
            new_pos_weight = torch.zeros((2048, EMBED_DIM), device=device)
            new_pos_weight[:old_pos_weight.shape[0], :] = old_pos_weight
            torch.nn.init.normal_(new_pos_weight[old_pos_weight.shape[0]:, :], mean=0.0, std=0.02)
            state_dict['pos_emb.weight'] = new_pos_weight

        model.load_state_dict(state_dict)
        print("✅ 成功加载并适配 Phys-GRPO 模型权重！")
    else:
        print(f"❌ 未找到模型权重: {model_path}，使用随机初始化进行演示。")

    target_dir = str(config.BENCHMARK_DIR)
    os.makedirs(target_dir, exist_ok=True)

    aigs = glob.glob(os.path.join(target_dir, "ctrl.aig"))
    if aigs:
        target = random.choice(aigs)
        optimize_with_guided_search(target, model, device)
    else:
        print(f"❌ 在 {target_dir} 中找不到测试用的 .aig 文件。")
