import torch, torch.nn as nn

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden

# --- toy training setup ---
text = "hello world hello there hello everyone"
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text])

seq_len = 8
def get_batch(bs=16):
    ix = torch.randint(0, len(data) - seq_len - 1, (bs,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y

model = CharLSTM(vocab_size=len(chars))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()

for step in range(500):
    x, y = get_batch()
    logits, _ = model(x)
    loss = criterion(logits.view(-1, len(chars)), y.view(-1))
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if step % 100 == 0:
        print(f"step {step}, loss {loss.item():.3f}")

# --- generate ---
@torch.no_grad()
def generate(model, start="h", length=30):
    model.eval()
    idx = torch.tensor([[stoi[start]]])
    hidden = None
    out = start
    for _ in range(length):
        logits, hidden = model(idx, hidden)
        probs = torch.softmax(logits[0, -1], dim=0)
        next_id = torch.multinomial(probs, 1).item()   # sampling — same idea as LLM decoding (Phase 4)
        out += itos[next_id]
        idx = torch.tensor([[next_id]])
    return out

print(generate(model))
