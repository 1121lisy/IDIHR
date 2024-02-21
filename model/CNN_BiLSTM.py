import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class CNN_BiLSTM(nn.Module):
    def __init__(self, embed_dim, num_classes, num_kernels=100, kernel_sizes=[3, 4, 5], hidden_dim=128, num_layers=2):
        super(CNN_BiLSTM, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_kernels, (k, embed_dim)) for k in kernel_sizes
        ])
        self.lstm = nn.LSTM(num_kernels , hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2 , num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, C, T, E]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [B, K, T]
        x = torch.cat((x[0],x[1],x[2]),dim=-1)
        x=x.transpose(1,2)


        x, _ = self.lstm(x)  # [B, T, 2H]
        x = x[:, -1, :]  # [B, 2H]
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)  # 全连接层




        return x