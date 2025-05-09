
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

# ===== STEP 1: Caricamento e Tokenizzazione =====
df = pd.read_csv("../data/processed/8withredSplit.csv", header=None)
df = df[:1000]


encoders = []
encoded_cols = []

for col in df.columns:
    le = LabelEncoder()
    encoded = le.fit_transform(df[col])
    encoders.append(le)
    encoded_cols.append(encoded)
    joblib.dump(le, f"encoder_col{col}.pkl")

# Converti in tensore
token_tensor = torch.tensor(list(zip(*encoded_cols)), dtype=torch.long)

# ===== STEP 2: DataLoader =====
BATCH_SIZE = 64
dataset = TensorDataset(token_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===== STEP 3: LSTM Autoencoder Model =====
class LSTMLogAutoencoder(nn.Module):
    def __init__(self, vocab_sizes, embedding_dim=16, hidden_dim=64):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vs, embedding_dim) for vs in vocab_sizes
        ])
        self.lstm_enc = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm_dec = nn.LSTM(hidden_dim, embedding_dim, batch_first=True)
        self.output_layers = nn.ModuleList([
            nn.Linear(embedding_dim, vs) for vs in vocab_sizes
        ])

    def forward(self, x):
        embedded = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        seq = torch.stack(embedded, dim=1)  # (batch, seq_len, emb_dim)

        _, (h_n, _) = self.lstm_enc(seq)
        dec_input = h_n.repeat(seq.size(1), 1, 1).transpose(0, 1)
        decoded, _ = self.lstm_dec(dec_input)

        outputs = []
        for i, out_layer in enumerate(self.output_layers):
            logits = out_layer(decoded[:, i])
            outputs.append(logits)

        return outputs

# ===== STEP 4: Training =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_sizes = [len(le.classes_) for le in encoders]
model = LSTMLogAutoencoder(vocab_sizes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

EPOCHS = 10
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    for batch in dataloader:
        batch = batch[0].to(device)
        outputs = model(batch)
        loss = sum(criterion(out, batch[:, i]) for i, out in enumerate(outputs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss/len(dataloader):.4f}")

# ===== STEP 5: Anomaly Detection =====
model.eval()
recon_losses = []

with torch.no_grad():
    for batch in DataLoader(dataset, batch_size=1):
        x = batch[0].to(device)
        outputs = model(x)
        loss = sum(criterion(out, x[:, i]) for i, out in enumerate(outputs))
        recon_losses.append(loss.item())

# ===== STEP 6: Calcolo soglia e anomalie =====
recon_losses = np.array(recon_losses)
mean = np.mean(recon_losses)
std = np.std(recon_losses)
threshold = mean + 3 * std  # soglia = μ + 3σ

print(f"Soglia anomalie: {threshold:.4f}")

# Righe anomale
anomalies = np.where(recon_losses > threshold)[0]
print(f"Anomalie trovate: {len(anomalies)}")
print("Indici righe anomale:", anomalies)
