import torch
from torch import nn
from transformers import AutoModel


class IDIHR_ra(nn.Module):
    def __init__(self, max_len, num_labels,dropout=0.1):
        super(IDIHR_ra, self).__init__()

        self.wobert = AutoModel.from_pretrained("junnyu/wobert_chinese_plus_base")

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.wobert.config.hidden_size+max_len, num_labels)
        self.a2 = SelfAttention(self.wobert.config.hidden_size)
        self.a3 = SelfAttention(max_len)

    def forward(self, x):
        w=x['wobert']
        m=x['mask']
        out_w = self.wobert(input_ids=w['input_ids'], token_type_ids=w['token_type_ids'],attention_mask=w['attention_mask'])

        pool_w = out_w.last_hidden_state[:, 0]
        pool_w = self.dropout(pool_w)
        x2= self.a2(pool_w)
        x3= self.a3(m)
        x=torch.cat((x2,x3),dim=1)
        x = self.fc(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_weights = torch.softmax(q @ k.transpose(-2, -1) / (self.key.in_features ** 0.5), dim=-1)
        output = attn_weights @ v

        return output