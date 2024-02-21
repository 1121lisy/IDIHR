from torch import nn
from transformers import AutoModel


class RoBertaClassifier(nn.Module):
    def __init__(self, bert_model_name, num_labels,dropout=0.1):
        super(RoBertaClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained('./'+bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, x):
        input_ids = x['input_ids']
        token_type_ids = x['token_type_ids']
        attention_mask = x['attention_mask']

        out = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask)
        pool = out.last_hidden_state[:, 0]
        dropout_output = self.dropout(pool)
        logits = self.linear(dropout_output)
        return logits
