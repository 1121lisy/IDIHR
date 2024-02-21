import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Transformer_BiLSTM(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads=4, hidden_dim=128, num_layers=2):
        super(Transformer_BiLSTM, self).__init__()

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads), num_layers=num_layers
        )
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.transformer(x)  # [B, T, E]
        x, _ = self.bilstm(x)  # [B, T, 2H]
        x = x[:, -1, :]  # [B, 2H]
        x = self.fc(x)  # [B, num_classes]
        return x