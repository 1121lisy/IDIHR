import torch
from torch import nn
from torch.nn import functional as F


class TextCNN(nn.Module):
    def __init__(self,  embed_dim, num_classes, num_kernels=100, kernel_sizes=[3, 4, 5]):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_kernels, (k, embed_dim)) for k in kernel_sizes
        ])
        self.fc = nn.Linear(num_kernels * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, C, T, E]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [B, K, T]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [B, K]
        x = torch.cat(x, 1)  # [B, K * len(kernel_sizes)]
        x = self.fc(x)
        return x
