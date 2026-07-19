import torch
from torch.utils.data import Dataset, DataLoader
from training_seq import inputs, targets

# __init__
# PyTorch models expect their inputs to be torch.Tensor objects, not NumPy arrays.
class CharDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

dataset = CharDataset(inputs, targets)
x, y = dataset[0]
print(x.shape)
print(y.shape)

batch_size = 64
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

for batch_inputs, batch_targets in train_loader:
    print(batch_inputs.shape)
    print(batch_targets.shape)
    break