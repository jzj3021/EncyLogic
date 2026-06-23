import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import math
from encylogic import config

EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 8
BATCH_SIZE = 10
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1000
SEQ_LEN = 25

try:
    from encylogic.actionspace import ACTION_SPACE1
except ImportError:
    print("❌ 错误: 找不到 actionspace.py 文件或 ACTION_SPACE1 变量。")
    exit(1)

stoi = {cmd: i for i, cmd in enumerate(ACTION_SPACE1)}
itos = {i: cmd for i, cmd in enumerate(ACTION_SPACE1)}

PAD_IDX = stoi["<pad>"]
BOS_IDX = stoi["<bos>"]
EOS_IDX = stoi["<eos>"]
VOCAB_SIZE = len(ACTION_SPACE1)

print(f"📋 词表大小: {VOCAB_SIZE}")
print(f"📋 动作空间: {ACTION_SPACE1}")

class AngelDataset(Dataset):
    def __init__(self, json_file):
        print(f"📂 正在加载数据: {json_file} ...")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ 读取 JSON 失败: {e}")
            data = []

        self.samples = []
        skipped_count = 0

        for item in data:
            seq = item.get('sequence', [])
            if not seq: continue

            try:
                token_ids = [BOS_IDX] + [stoi[op] for op in seq] + [EOS_IDX]
            except KeyError as e:
                skipped_count += 1
                continue

            if len(token_ids) < SEQ_LEN + 2:
                token_ids += [PAD_IDX] * (SEQ_LEN + 2 - len(token_ids))
            else:
                token_ids = token_ids[:SEQ_LEN + 2]

            self.samples.append(torch.tensor(token_ids, dtype=torch.long))

        print(f"✅ 数据加载完成: 有效样本 {len(self.samples)} 条 (跳过 {skipped_count} 条异常数据)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        return tokens[:-1], tokens[1:]

class SeqFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, EMBED_DIM))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=512,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=NUM_LAYERS)

        self.fc_out = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, x):
        seq_len = x.size(1)

        emb = self.embedding(x) + self.pos_encoder[:, :seq_len, :]

        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        output = self.transformer_decoder(emb, memory=emb, tgt_mask=mask, memory_mask=mask)

        return self.fc_out(output)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    data_path = str(config.DATA_DIR / "top50_best_sequences_final.json")
    if not list(data_path):
        print(f"❌ 找不到文件: {data_path}")
        return

    dataset = AngelDataset(data_path)
    if len(dataset) == 0:
        print("❌ 数据集为空，无法训练")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SeqFormer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    print("开始训练 SeqFormer...")
    model.train()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch+1) % 1 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f}")

    save_path = str(config.SFT_CHECKPOINT)
    torch.save(model.state_dict(), save_path)
    print(f"💾 模型已保存到 {save_path}")

if __name__ == "__main__":
    train()
