import torch
import torch.nn as nn

# we RETURN Logits
# CrossEntropyLoss expects raw logits and applies the appropriate normalization internally. 
# This is more numerically stable than applying softmax yourself.
class CharLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        hidden_size=256,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(                  ## -> LSTM
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        logits = self.fc(output)

        return logits



###   GRU version of the model
class CharGRU(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        hidden_size=256,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.gru(x)
        logits = self.fc(output)
        return logits