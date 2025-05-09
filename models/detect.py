import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


# Definizione del modello LSTM (già fornito nel codice precedente)
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


# Funzione per caricare modello e punteggi di anomalia
def load_data_and_model():
    # Carica il modello
    model = LSTMAutoencoder(embedding_dim=128, hidden_dim=64, sequence_len=20)  # Usa le tue dimensioni
    model.load_state_dict(torch.load("lstm_autoencoder.pth"))
    model.eval()

    # Carica i punteggi di anomalia
    data = torch.load("anomaly_scores_with_timestamps.pt")
    timestamps = data['timestamps']
    scores = data['scores']

    return model, timestamps, scores


# Funzione per visualizzare i punteggi di anomalia
def plot_anomaly_scores(timestamps, scores):
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, scores, label="Anomaly Scores")
    plt.title("Punteggi di Anomalia (Errore di Ricostruzione)")
    plt.xlabel("Timestamp")
    plt.ylabel("Anomaly Score")
    plt.grid(True)
    plt.show()


# Funzione per calcolare e definire una soglia di anomalia
def define_anomaly_threshold(scores):
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    # Soglia di anomalia come media + 3 deviazioni standard
    threshold = mean_score + 3 * std_score
    print(f"Soglia per anomalie: {threshold}")

    # Trova gli indici con punteggio di anomalia superiore alla soglia
    anomalous_indexes = [i for i, score in enumerate(scores) if score > threshold]

    return anomalous_indexes, threshold


# Funzione per salvare i timestamp anomali
def save_anomalous_timestamps(timestamps, anomalous_indexes):
    anomalous_timestamps = [timestamps[i] for i in anomalous_indexes]

    print(f"Anomalie trovate: {len(anomalous_indexes)}")
    print("Timestamp con anomalie:")
    for timestamp in anomalous_timestamps:
        print(timestamp)

    # Salva i timestamp anomali in un file
    with open("anomalous_timestamps.txt", "w") as f:
        for timestamp in anomalous_timestamps:
            f.write(str(timestamp) + "\n")

    print("Anomalie salvate in 'anomalous_timestamps.txt'")


# Funzione principale
def main():
    # Carica il modello e i punteggi di anomalia
    model, timestamps, scores = load_data_and_model()

    # Visualizza i punteggi di anomalia
    plot_anomaly_scores(timestamps, scores)

    # Definisci e applica la soglia di anomalia
    anomalous_indexes, threshold = define_anomaly_threshold(scores)

    # Salva i timestamp anomali in un file
    save_anomalous_timestamps(timestamps, anomalous_indexes)


# Esegui il programma
if __name__ == "__main__":
    main()
