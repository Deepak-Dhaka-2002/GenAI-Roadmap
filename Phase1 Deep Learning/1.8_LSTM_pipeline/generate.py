import torch
from models import CharLSTM , CharGRU
from test_encoding import char_to_idx, idx_to_char, vocab_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CharLSTM(
#     vocab_size=vocab_size,
#     embedding_dim=64,
#     hidden_size=256,
#     num_layers=2,
#     dropout=0.2,
# )
# model.load_state_dict(torch.load("char_lstm.pth", map_location=device))

model = CharGRU(
    vocab_size=vocab_size,
    embedding_dim=64,
    hidden_size=256,
    num_layers=2,
    dropout=0.2
)
model.load_state_dict( torch.load("char_gru.pth", map_location=device))

model.to(device)
model.eval()

seed_text = "BAPTISTA:"
input_ids = [char_to_idx[c] for c in seed_text]
input_tensor = torch.tensor(
    input_ids,
    dtype=torch.long
).unsqueeze(0).to(device)

generated_text = seed_text
generate_length = 500

def generate_text(seed_text, length=500, temperature=1.0):
    model.eval()
    input_ids = [char_to_idx[c] for c in seed_text]

    input_tensor = torch.tensor(
        input_ids,
        dtype=torch.long
    ).unsqueeze(0).to(device)
    generated = seed_text
    with torch.no_grad():
        for _ in range(length):
            outputs = model(input_tensor)

            logits = outputs[:, -1, :]
            logits = logits / temperature
            probabilities = torch.softmax(logits, dim=-1)
            next_char = torch.multinomial(
                probabilities,
                num_samples=1
            )
            predicted_char = idx_to_char[next_char.item()]

            generated += predicted_char
            input_tensor = torch.cat(
                [input_tensor, next_char],
                dim=1
            )
            if input_tensor.size(1) > 100:
                input_tensor = input_tensor[:, -100:]

    return generated

def generate_greedy(seed_text, length=500):
    model.eval()
    input_ids = [char_to_idx[c] for c in seed_text]
    input_tensor = torch.tensor(
        input_ids,
        dtype=torch.long
    ).unsqueeze(0).to(device)
    generated = seed_text
    with torch.no_grad():
        for _ in range(length):
            outputs = model(input_tensor)
            logits = outputs[:, -1, :]

            next_char = torch.argmax(
                logits,
                dim=-1,
                keepdim=True
            )
            generated += idx_to_char[next_char.item()]

            input_tensor = torch.cat(
                [input_tensor, next_char],
                dim=1
            )
            if input_tensor.size(1) > 100:
                input_tensor = input_tensor[:, -100:]

    return generated



print("=" * 60)
print("GREEDY DECODING")
print("=" * 60)
print(generate_greedy(seed_text))

print("=" * 60)
print("TEMPERATURE = 0.5")
print("=" * 60)
print(generate_text(seed_text, temperature=0.5))

print("=" * 60)
print("TEMPERATURE = 1.0")
print("=" * 60)
print(generate_text(seed_text, temperature=1.0))

print("=" * 60)
print("TEMPERATURE = 1.5")
print("=" * 60)
print(generate_text(seed_text, temperature=1.5))

# REPORT WE OBSERVE FROM THE GENERATED TEXTS:
# | Method            | Characteristics                                                                                                                |
# | ----------------- | ------------------------------------------------------------------------------------------------------------------------------ |
# | Greedy            | Produced the most coherent and deterministic text but often repeated common patterns and closely followed the training corpus. |
# | Temperature = 0.5 | Slightly more varied than greedy while remaining highly coherent.                                                              |
# | Temperature = 1.0 | Best balance between diversity and readability, generating more original continuations.                                        |
# | Temperature = 1.5 | Most diverse output, but grammar and coherence degraded due to increased randomness.                                           |

# | Method            | Coherence | Diversity | Comments                                 |
# | ----------------- | --------- | --------- | ---------------------------------------- |
# | Greedy            | ⭐⭐⭐⭐⭐     | ⭐         | Most accurate but repetitive             |
# | Temperature = 0.5 | ⭐⭐⭐⭐⭐     | ⭐⭐        | Similar to greedy with slight randomness |
# | Temperature = 1.0 | ⭐⭐⭐⭐☆     | ⭐⭐⭐⭐      | Best balance of quality and diversity    |
# | Temperature = 1.5 | ⭐⭐⭐       | ⭐⭐⭐⭐⭐     | Highly creative but less coherent        |
