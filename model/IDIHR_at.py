import torch
from torch import nn
from transformers import AutoModel


class IDIHR_at(nn.Module):
    def __init__(self, max_len,bert_model_name, num_labels,dropout=0.1):
        super(IDIHR_at, self).__init__()

        self.wobert = AutoModel.from_pretrained("junnyu/wobert_chinese_plus_base")
        self.roberta = AutoModel.from_pretrained('./'+bert_model_name)
        """for param in self.wobert.parameters():
            param.requires_grad = False
        for param in self.roberta.parameters():
            param.requires_grad = False"""
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.roberta.config.hidden_size+self.wobert.config.hidden_size+max_len, num_labels)


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

        x=torch.cat((pool_r,pool_w,m),dim=1)
        x = self.fc(x)
        return x

