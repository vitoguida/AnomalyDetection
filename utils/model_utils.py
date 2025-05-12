import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data_utils import categorical_cols

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

def train_model(model, dataloader, device, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = sum(F.cross_entropy(outputs[col], y_batch[:, i]) for i, col in enumerate(categorical_cols))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return epoch_losses

def plot_training_loss(losses):
    import matplotlib.pyplot as plt
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

def compute_anomaly_score(model, input_seq, true_event, device):
    model.eval()
    input_seq, true_event = input_seq.to(device), true_event.to(device)
    with torch.no_grad():
        output = model(input_seq)
    total_loss = sum(
        F.cross_entropy(output[col], true_event[:, i], reduction='none').item()
        for i, col in enumerate(categorical_cols)
    )
    return total_loss

def is_event_anomalous(df, model, index, seq_len, threshold, device):
    if index < seq_len:
        raise ValueError("Indice troppo piccolo per creare una sequenza.")
    sequence = df.iloc[index - seq_len:index]
    target = df.iloc[index]

    x_seq = torch.tensor(sequence[categorical_cols].astype(int).values, dtype=torch.long).unsqueeze(0)
    y_target = torch.tensor(target[categorical_cols].astype(int).values, dtype=torch.long).unsqueeze(0)

    score = compute_anomaly_score(model, x_seq, y_target, device)
    return score, score > threshold

def find_anomalies(df):
    return df[df['is_redteam'].astype(str) == '1'].index.tolist()
