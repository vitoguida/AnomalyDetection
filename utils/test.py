import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
import matplotlib.pyplot as plt

# ===========================
# PARAMS
# ===========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # per multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
csv_path = "../data/processed/8withredSplit.csv"
seq_len = 10
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===========================
# LOAD DATA
# ===========================
df = pd.read_csv(csv_path, header=None, dtype=str, low_memory=False)
#df = df[3200000:6855000]
df = df[2500000:2580000]
df = df.reset_index(drop=True)
df.columns = ["time", "src_user", "src_domain", "dst_user", "dst_domain",
              "src_comp", "dst_comp", "auth_type", "logon_type", "orientation",
              "success", "is_redteam"]

# ===========================
# ENCODING CATEGORICAL FEATURES
# ===========================
categorical_cols = ["src_user", "src_domain", "dst_user", "dst_domain",
                    "src_comp", "dst_comp", "auth_type", "logon_type",
                    "orientation", "success"]

encoders = {}
for col in categorical_cols:
    df[col] = df[col].astype(str).fillna("UNK")
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ===========================
# DATASET CLASS
# ===========================
class AuthSequenceDataset(Dataset):
    def __init__(self, df, seq_len):
        self.seq_len = seq_len
        self.data = []
        for i in range(len(df) - seq_len):
            seq = df.iloc[i:i+seq_len]
            target = df.iloc[i+seq_len]
            self.data.append((seq[categorical_cols].values, target[categorical_cols].values))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_seq, y_target = self.data[idx]
        x_seq = torch.tensor(x_seq, dtype=torch.long)
        y_target = torch.tensor(y_target.astype(int), dtype=torch.long)
        return x_seq, y_target

# ===========================
# MODEL
# ===========================
class LSTMAnomalyModel(nn.Module):
    def __init__(self, num_features, vocab_sizes, emb_dim=32, hidden_dim=64):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim) for vocab_size in vocab_sizes
        ])
        self.lstm = nn.LSTM(input_size=emb_dim * num_features, hidden_size=hidden_dim, batch_first=True)
        self.output_layers = nn.ModuleDict({
            col: nn.Linear(hidden_dim, vocab_sizes[i])
            for i, col in enumerate(categorical_cols)
        })

    def forward(self, x):
        embedded = [self.embeddings[i](x[:, :, i]) for i in range(len(categorical_cols))]
        x_embed = torch.cat(embedded, dim=-1)
        _, (h_n, _) = self.lstm(x_embed)
        h = h_n[-1]
        return {col: self.output_layers[col](h) for col in categorical_cols}

# ===========================
# TRAINING FUNCTION
# ===========================


def train_model(model, dataloader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    epoch_losses = []  # Per tracciare la loss media per ogni epoca

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = 0
            for i, col in enumerate(categorical_cols):
                loss += F.cross_entropy(outputs[col], y_batch[:, i])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # === PLOT LOSS ===
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', color='b')
    plt.title("Andamento della Loss durante il Training")
    plt.xlabel("Epoca")
    plt.ylabel("Loss media")
    plt.grid(True)
    plt.show()


# ===========================
# ANOMALY SCORING
# ===========================
def compute_anomaly_score(model, input_seq, true_event, device ):
    model.eval()
    input_seq = input_seq.to(device)
    true_event = true_event.to(device)
    with torch.no_grad():
        output = model(input_seq)
    total_loss = 0.0
    for i, col in enumerate(categorical_cols):
        pred_logits = output[col]
        target_idx = true_event[:, i]
        loss = F.cross_entropy(pred_logits, target_idx, reduction='none')
        total_loss += loss.item()
    return total_loss


def is_event_anomalous(df, model, index, seq_len, threshold, device ):
    if index < seq_len:
        raise ValueError("Indice troppo piccolo per creare una sequenza precedente.")

    sequence = df.iloc[index - seq_len:index]
    target = df.iloc[index]

    x_seq = torch.tensor(sequence[categorical_cols].astype(int).values, dtype=torch.long).unsqueeze(0)
    y_values = target[categorical_cols].astype(int).values
    y_target = torch.tensor(y_values, dtype=torch.long).unsqueeze(0)

    score = compute_anomaly_score(model, x_seq, y_target, device)
    return score, score > threshold

def find_anomalies(df):
    # Assicura che la colonna sia trattata come stringa, poi filtra le righe dove il valore è esattamente '1'
    anomaly_indices = df[df['is_redteam'].astype(str) == '1'].index.tolist()
    return anomaly_indices


# ===========================
# RUN TRAINING
# ===========================
dataset = AuthSequenceDataset(df, seq_len=seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

vocab_sizes = [len(encoders[col].classes_) for col in categorical_cols]
model = LSTMAnomalyModel(num_features=len(categorical_cols), vocab_sizes=vocab_sizes)

train_model(model, dataloader, epochs=30)

# ===========================
# EXAMPLE ANOMALY CHECK
# ===========================
threshold = 10.0
anomaly_indices = find_anomalies(df)

print(f"Totale righe marcate come anomalie (ground truth): {len(anomaly_indices)}")

# Per contare quante sono state classificate come anomale dal modello
true_detected = 0


for idx_to_check in anomaly_indices:
    try:
        score, is_anom = is_event_anomalous(df, model, idx_to_check, seq_len, threshold, device)
        print(f"Anomaly Score: {score:.2f} -> {'ANOMALO' if is_anom else 'normale'}")
        if is_anom:
            true_detected += 1
    except ValueError:
        continue  # Ignora gli indici troppo piccoli

print(f"Anomalie classificate come anomale dal modello: {true_detected}")
print(f"Percentuale rilevate correttamente: {100 * true_detected / len(anomaly_indices):.2f}%")


"""idx_to_check = 50
score, is_anom = is_event_anomalous(df, model, idx_to_check, seq_len, threshold, device)
print(f"[Index {idx_to_check}] Anomaly Score: {score:.2f} -> {'ANOMALO' if is_anom else 'normale'}")

idx_to_check = 230
score, is_anom = is_event_anomalous(df, model, idx_to_check, seq_len, threshold, device)
print(f"[Index {idx_to_check}] Anomaly Score: {score:.2f} -> {'ANOMALO' if is_anom else 'normale'}")

idx_to_check = 123
score, is_anom = is_event_anomalous(df, model, idx_to_check, seq_len, threshold, device)
print(f"[Index {idx_to_check}] Anomaly Score: {score:.2f} -> {'ANOMALO' if is_anom else 'normale'}")"""