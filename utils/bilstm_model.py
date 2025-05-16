import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import categorical_cols


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, lstm_outputs):
        # Calcoliamo i pesi di attenzione come un prodotto scalare tra gli output LSTM e i pesi di attenzione
        attention_scores = torch.matmul(lstm_outputs, self.attention_weights)
        attention_weights = F.softmax(attention_scores, dim=1)
        # Applichiamo i pesi di attenzione agli output LSTM per ottenere una rappresentazione ponderata
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * lstm_outputs, dim=1)
        return context_vector


class BiLSTMAnomalyModel(nn.Module):
    def __init__(self, num_features, vocab_sizes, emb_dim=32, hidden_dim=64, lstm_dropout=0.3, DROP=False):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim) for vocab_size in vocab_sizes
        ])
        self.DROP = DROP
        self.hidden_dim = hidden_dim

        if DROP:
            self.lstm = nn.LSTM(
                input_size=emb_dim * num_features,
                hidden_size=hidden_dim,
                batch_first=True,
                dropout=lstm_dropout,  # Dropout tra i layer dell’LSTM
                num_layers=2,
                bidirectional=True  # LSTM bidirezionale
            )
            self.dropout = nn.Dropout(p=lstm_dropout)
        else:
            self.lstm = nn.LSTM(
                input_size=emb_dim * num_features,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True  # LSTM bidirezionale
            )

        self.attention = Attention(hidden_dim * 2)  # Due volte il hidden_dim perché bidirezionale
        self.output_layers = nn.ModuleDict({
            col: nn.Linear(hidden_dim * 2, vocab_sizes[i])  # Due volte hidden_dim per la bidirezionalità
            for i, col in enumerate(categorical_cols)
        })

    def forward(self, x):
        # Embedding dei dati
        embedded = [self.embeddings[i](x[:, :, i]) for i in range(len(categorical_cols))]
        x_embed = torch.cat(embedded, dim=-1)

        # Passaggio attraverso l'LSTM
        lstm_out, (h_n, c_n) = self.lstm(x_embed)

        # Calcolo del contesto attraverso il meccanismo di attenzione
        context_vector = self.attention(lstm_out)

        if self.DROP:
            context_vector = self.dropout(context_vector)

        # Output finale
        return {col: self.output_layers[col](context_vector) for col in categorical_cols}



