import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

# ======== MODELLO ========
class LSTMAutoencoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, sequence_len):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=embedding_dim, batch_first=True)
        self.sequence_len = sequence_len
        self.hidden_dim = hidden_dim

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        decoder_input = hidden.repeat(self.sequence_len, 1, 1).permute(1, 0, 2)
        decoded_output, _ = self.decoder(decoder_input)
        return decoded_output

# ======== DATASET ========
class LogDataset(Dataset):
    def __init__(self, data, sequence_length=20):
        self.sequence_length = sequence_length
        self.sequences = []
        self.timestamps = []

        # Appiattisci tutto preservando timestamp
        flattened = []
        for timestamp, embedding in data:
            for vec in embedding:
                flattened.append((timestamp, vec))

        for i in range(len(flattened) - sequence_length):
            seq_embeddings = [flattened[j][1] for j in range(i, i + sequence_length)]
            seq_timestamp = flattened[i][0]  # timestamp iniziale
            self.sequences.append(torch.stack(seq_embeddings))
            self.timestamps.append(seq_timestamp)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.timestamps[idx]

# ======== ALLENAMENTO ========
def train_autoencoder(model, dataloader, num_epochs=10, lr=1e-3, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch, _ in dataloader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")

# ======== ANOMALY SCORES ========
def compute_reconstruction_errors(model, dataloader, device='cuda'):
    model.eval()
    errors = []
    timestamps = []

    with torch.no_grad():
        for batch, batch_timestamps in dataloader:
            batch = batch.to(device)
            output = model(batch)
            loss = nn.functional.mse_loss(output, batch, reduction='none')
            loss_per_sequence = loss.mean(dim=(1, 2))  # errore medio per sequenza
            errors.extend(loss_per_sequence.cpu().numpy())
            timestamps.extend(batch_timestamps)

    return timestamps, errors

# ======== MAIN ========
if __name__ == "__main__":
    # === CONFIGURAZIONE ===
    EMBEDDING_FILE = "../data/processed/embeddings/embedding_batch_0.pt"
    SEQUENCE_LENGTH = 20
    HIDDEN_DIM = 64
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Caricamento embedding...")
    data = torch.load(EMBEDDING_FILE)
    data = data[:250000]
    print("Preparazione sequenze...")
    dataset = LogDataset(data, sequence_length=SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    embedding_dim = dataset[0][0].shape[1]  # dataset[idx] → (sequence, timestamp)

    print("Creazione modello...")
    model = LSTMAutoencoder(embedding_dim=embedding_dim, hidden_dim=HIDDEN_DIM, sequence_len=SEQUENCE_LENGTH)

    print("Inizio training...")
    train_autoencoder(model, dataloader, num_epochs=NUM_EPOCHS, device=DEVICE)

    print("Calcolo errori di ricostruzione...")
    timestamps, errors = compute_reconstruction_errors(model, dataloader, device=DEVICE)

    print("Salvataggio risultati...")
    torch.save(model.state_dict(), "lstm_autoencoder.pth")
    torch.save({"timestamps": timestamps, "scores": errors}, "anomaly_scores_with_timestamps.pt")

    print("Completato.")
