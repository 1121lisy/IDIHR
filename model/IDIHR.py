import torch
from torch import nn
from transformers import AutoModel


class IDIHR(nn.Module):
    def __init__(self, max_len,bert_model_name, num_labels,dropout=0.1):
        super(IDIHR, self).__init__()

        self.wobert = AutoModel.from_pretrained("junnyu/wobert_chinese_plus_base")
        self.roberta = AutoModel.from_pretrained('./'+bert_model_name)
        """for param in self.wobert.parameters():
            param.requires_grad = False
        for param in self.roberta.parameters():
            param.requires_grad = False"""
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.roberta.config.hidden_size+self.wobert.config.hidden_size+max_len, num_labels)
        self.a1 = SelfAttention(self.roberta.config.hidden_size)
        self.a2 = SelfAttention(self.wobert.config.hidden_size)
        self.a3 = SelfAttention(max_len)

    def forward(self, x):
        r=x['roberta']
        w=x['wobert']
        m=x['mask']
        out_r = self.roberta(input_ids=r['input_ids'], token_type_ids=r['token_type_ids'],attention_mask=r['attention_mask'])
        out_w = self.wobert(input_ids=w['input_ids'], token_type_ids=w['token_type_ids'],attention_mask=w['attention_mask'])
        pool_r = out_r.last_hidden_state[:, 0]
        pool_r = self.dropout(pool_r)
        pool_w = out_w.last_hidden_state[:, 0]
        pool_w = self.dropout(pool_w)
        x1= self.a1(pool_r)
        x2= self.a2(pool_w)
        x3= self.a3(m)
        x=torch.cat((x1,x2,x3),dim=1)
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