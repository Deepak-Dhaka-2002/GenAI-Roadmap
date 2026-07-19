import torch
import torch.nn as nn
import torch.optim as optim
import json
from models import CharLSTM , CharGRU
from dataloader import train_loader
from test_model import vocab_size
import matplotlib.pyplot as plt
## for both models change 2 things model assign (line 16) , model save at end

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(device)

# model = CharLSTM(vocab_size=60)
model = CharGRU(vocab_size=60)  # Use the GRU model instead of LSTM
model.to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 10
train_losses = []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        outputs = outputs.reshape(-1, vocab_size)
        targets = targets.reshape(-1)

        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5
        )

        optimizer.step()
        epoch_loss += loss.item()
    average_loss = epoch_loss / len(train_loader)
    train_losses.append(average_loss)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {average_loss:.4f}")


# torch.save(
#     model.state_dict(),
#     "D:\\Courses\\Vscode\\Agentic_AI\\1.8_LSTM_pipeline\\char_lstm.pth"
# )
torch.save(
    model.state_dict(),
    "D:\\Courses\\Vscode\\Agentic_AI\\1.8_LSTM_pipeline\\char_gru.pth"
)
print("Model saved.")

#save loses for plotting
with open("D:\\Courses\\Vscode\\Agentic_AI\\1.8_LSTM_pipeline\\train_losses.json", "w") as f:
    json.dump(train_losses, f)

with open("D:\\Courses\\Vscode\\Agentic_AI\\1.8_LSTM_pipeline\\train_losses.json", "r") as f:
    train_losses = json.load(f)

plt.figure(figsize=(8,5))
plt.plot(range(1, len(train_losses)+1), train_losses, marker="o")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()