import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class BiGRU_Attention(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads=8, hidden_dim=128, num_layers=2):
        super(BiGRU_Attention, self).__init__()

        self.bigru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        encoder_layers = TransformerEncoderLayer(hidden_dim * 2, num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x, _ = self.bigru(x)  # [B, T, 2H]
        x = self.transformer_encoder(x)  # [B, T, 2H]
        x = x[:, -1, :]  # [B, 2H]
        x = self.fc(x)  # [B, num_classes]
        return x