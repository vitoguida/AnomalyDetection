import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data_utils import categorical_cols
#

class LSTMAnomalyModel(nn.Module):
    def __init__(self, num_features, vocab_sizes, emb_dim=32, hidden_dim=64, lstm_dropout=0.3,DROP=False):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim) for vocab_size in vocab_sizes
        ])
        self.DROP = DROP
        if DROP == True:
            self.lstm = nn.LSTM(
                input_size=emb_dim * num_features,
                hidden_size=hidden_dim,
                batch_first=True,
                dropout=lstm_dropout,  # Dropout tra i layer dell’LSTM (se >1 layer)
                num_layers=1  # Per attivare il dropout servono ≥2 layer
            )
            self.dropout = nn.Dropout(p=lstm_dropout)  # Dropout esplicito dopo LSTM
            self.output_layers = nn.ModuleDict({
                col: nn.Linear(hidden_dim, vocab_sizes[i])
                for i, col in enumerate(categorical_cols)
            })

        else :
            self.lstm = nn.LSTM(input_size=emb_dim * num_features, hidden_size=hidden_dim, batch_first=True)
            self.output_layers = nn.ModuleDict({
                col: nn.Linear(hidden_dim, vocab_sizes[i])
                for i, col in enumerate(categorical_cols)
            })


    def forward(self, x):  # QUI è dove forward deve stare
        embedded = [self.embeddings[i](x[:, :, i]) for i in range(len(categorical_cols))]
        x_embed = torch.cat(embedded, dim=-1)
        _, (h_n, _) = self.lstm(x_embed)
        h = h_n[-1]
        if self.DROP == True:
            h = self.dropout(h)
        return {col: self.output_layers[col](h) for col in categorical_cols}


