import torch
from torch import nn
from transformers import AutoModel


class IDIHR_hr(nn.Module):
    def __init__(self, max_len,bert_model_name, num_labels,dropout=0.1):
        super(IDIHR_hr, self).__init__()


        self.roberta = AutoModel.from_pretrained('./'+bert_model_name)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.roberta.config.hidden_size+max_len, num_labels)
        self.a1 = SelfAttention(self.roberta.config.hidden_size)

        self.a3 = SelfAttention(max_len)

    def forward(self, x):
        r=x['roberta']
        m=x['mask']
        out_r = self.roberta(input_ids=r['input_ids'], token_type_ids=r['token_type_ids'],attention_mask=r['attention_mask'])
        pool_r = out_r.last_hidden_state[:, 0]
        pool_r = self.dropout(pool_r)

        x1= self.a1(pool_r)
        x3= self.a3(m)
        x=torch.cat((x1,x3),dim=1)
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