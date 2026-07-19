import torch
from models import CharLSTM

vocab_size = 60
model = CharLSTM(vocab_size)
x = torch.randint(0, vocab_size, (64, 100))
output = model(x)

print(output.shape)

total_params = sum(p.numel() for p in model.parameters())
# See how many trainable weights the model has, which is useful when we later compare it with the GRU version.
print("Total Parameters:", total_params)