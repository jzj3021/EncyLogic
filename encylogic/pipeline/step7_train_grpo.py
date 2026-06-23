import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import glob
import random
import numpy as np
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from encylogic import config

EMBED_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 6
SEQ_LEN = 25
GROUP_SIZE = 32
LEARNING_RATE = 5e-6
KL_COEFF = 0.02
EPOCHS = 120

MAX_WORKERS = 32

session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS))

try:
    from encylogic.actionspace import ACTION_SPACE1
except ImportError:
    print("❌ 错误: 找不到 actionspace.py")
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
        self.pos_emb = nn.Embedding(100, EMBED_DIM)

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

def evaluate_sequence_via_api(args):
    """
    将生成的序列发送给 OpenROAD 容器，获取真实的物理 Reward (负的 Delay)
    """
    idx, design_name, seq_idxs = args

    cmds = [itos.get(int(val.item() if isinstance(val, (torch.Tensor, np.generic)) else val), "")
            for val in seq_idxs]
    cmds = [cmd for cmd in cmds if cmd and cmd not in ["<bos>", "<eos>", "<pad>"]]

    if not cmds:
        return -9999.0

    payload = {
        "design": design_name,
        "sequence": cmds
    }

    try:
        response = session.post(config.API_URL, json=payload, timeout=(10, 2000))
        result = response.json()

        if result.get("status") == "success":
            delay = float(result.get("delay", -9999.0))

            if delay == -9999.0:
                return -9999.0

            print(f"  ✅ [后端捷报] {design_name} 物理 Delay: {delay} ns")

            return -delay
        else:
            return -9999.0

    except Exception as e:
        print(f"\n❌ [API 致命错误] 无法连接到 Docker: {e}")
        return -9999.0

def run_env_batch(circuit_paths, sequences):
    """
    并发调用 OpenROAD 环境，获取整组 (Group Size) 的物理 Reward
    """
    tasks = []
    for i, (circ_path, seq_idxs) in enumerate(zip(circuit_paths, sequences)):
        design_name = os.path.splitext(os.path.basename(circ_path))[0]
        tasks.append((i, design_name, seq_idxs))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(evaluate_sequence_via_api, tasks))

    return torch.tensor(results, dtype=torch.float32)

def train_phys_grpo():
    global_start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动 Phys-GRPO (优化目标: Delay) | 设备: {device} | MAX_WORKERS: {MAX_WORKERS}")

    model = SeqFormer().to(device)
    pretrained_path = str(config.SFT_CHECKPOINT)
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
        print("✅ 加载 Phase 1 SFT 模型成功！")
    else:
        print("⚠️ 未找到预训练模型，使用随机初始化")

    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    train_aigs = glob.glob(os.path.join(str(config.AIG_DIR), "*.aig"))
    if not train_aigs: train_aigs = glob.glob("*.aig")
    print(f"📚 训练电路数量: {len(train_aigs)}")
    if len(train_aigs) == 0: return

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        total_reward = 0
        random.shuffle(train_aigs)

        for step, circuit_path in enumerate(train_aigs):
            batch_circuits = [circuit_path] * GROUP_SIZE
            curr_input = torch.full((GROUP_SIZE, 1), BOS_IDX, dtype=torch.long, device=device)

            model.eval()
            with torch.no_grad():
                for _ in range(SEQ_LEN):
                    logits = model(curr_input)
                    next_token_logits = logits[:, -1, :]

                    next_token_logits[:, PAD_IDX] = -float('inf')
                    next_token_logits[:, BOS_IDX] = -float('inf')

                    dist = Categorical(logits=next_token_logits)
                    action = dist.sample()
                    curr_input = torch.cat([curr_input, action.unsqueeze(1)], dim=1)

            actions = curr_input[:, 1:]

            rewards = run_env_batch(batch_circuits, actions.cpu().numpy()).to(device)
            mean_r, std_r = rewards.mean(), rewards.std() + 1e-8

            advantages = ((rewards - mean_r) / std_r).unsqueeze(1)

            model.train()

            policy_logits = model(curr_input[:, :-1])
            with torch.no_grad():
                ref_logits = ref_model(curr_input[:, :-1])

            inf_mask = torch.zeros_like(policy_logits)
            inf_mask[:, :, PAD_IDX] = -float('inf')
            inf_mask[:, :, BOS_IDX] = -float('inf')

            policy_logits = policy_logits + inf_mask
            ref_logits = ref_logits + inf_mask

            policy_dist = Categorical(logits=policy_logits)
            log_probs = policy_dist.log_prob(actions)

            ref_dist = Categorical(logits=ref_logits)
            ref_log_probs = ref_dist.log_prob(actions)

            log_ratio = ref_log_probs - log_probs
            kl_div = torch.exp(log_ratio) - log_ratio - 1.0
            kl_loss = kl_div.mean()

            pg_loss = -(log_probs * advantages).mean()

            loss = pg_loss + KL_COEFF * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_reward += mean_r.item()

            if (step+1) % 5 == 0 or step == 0:
                avg_delay = -mean_r.item()
                best_delay = -rewards.max().item()
                print(f"  ⚡ 电路 {os.path.basename(circuit_path)[:15]:<15} | Avg Delay: {avg_delay:7.2f}ns | Best Delay: {best_delay:7.2f}ns | KL: {kl_loss.item():.4f}")

        avg_epoch_reward = total_reward / max(len(train_aigs), 1)
        epoch_time = time.time() - epoch_start_time
        print(f"🏆 Epoch {epoch+1}/{EPOCHS} 结束 | 平均 Delay: {-avg_epoch_reward:.2f}ns | 本轮耗时: {epoch_time:.2f}s")

        save_dir = str(config.MODEL_DIR)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "seqformer_phys_grpo_new.pth"))

    total_time = time.time() - global_start_time
    hours, rem = divmod(total_time, 3600)
    mins, secs = divmod(rem, 60)
    print(f"\n🎉 物理驱动 RL 训练圆满完成！总耗时: {int(hours)}小时 {int(mins)}分钟 {secs:.2f}秒")

if __name__ == "__main__":
    train_phys_grpo()
