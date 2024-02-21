import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self,  embedding_dim, hidden_dim, output_dim, n_layers=1,
                 bidirectional=True, dropout=0.5):
        super().__init__()

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           batch_first=True,
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        output, (hidden, cell) = self.rnn(x)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.fc(hidden.squeeze(0))