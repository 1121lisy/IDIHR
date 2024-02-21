import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class BiLSTM_MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads=8, hidden_dim=128, num_layers=2):
        super(BiLSTM_MultiheadAttention, self).__init__()

        self.bilstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        encoder_layers = TransformerEncoderLayer(hidden_dim * 2, num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_dim , num_classes)

    def forward(self, x):
        x, _ = self.bilstm(x)  # [B, T, H]
        x = self.transformer_encoder(x)  # [B, T, H]
        x = x[:, -1, :]  # [B, H]
        x = self.fc(x)  # [B, num_classes]
        return x