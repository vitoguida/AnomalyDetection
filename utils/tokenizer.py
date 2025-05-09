import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn


class LogAnomalyModel(nn.Module):
    def __init__(self, n_unique_per_column, embedding_dim=16):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=n, embedding_dim=embedding_dim)
            for n in n_unique_per_column
        ])
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * len(n_unique_per_column), 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # o numero di classi
        )

    def forward(self, x):
        embedded = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embedded, dim=1)
        return self.fc(x)


# Carica i dati
df = pd.read_csv("../data/processed/8withredSplit.csv", header=None)
df = df[:1000]

# Costruisci encoder per ogni colonna
encoders = []
encoded_columns = []

for col in df.columns:
    le = LabelEncoder()
    encoded_col = le.fit_transform(df[col])
    encoders.append(le)
    encoded_columns.append(encoded_col)

# Stack in tensore
X = torch.tensor(list(zip(*encoded_columns)))  # shape: (n_samples, n_features)



n_unique_per_column = [len(le.classes_) for le in encoders]
model = LogAnomalyModel(n_unique_per_column)
