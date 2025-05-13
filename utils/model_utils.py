import torch
import torch.nn.functional as F
from data_utils import categorical_cols

import numpy as np

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

"""def compute_anomaly_score(model, input_seq, true_event, device):
    model.eval()
    input_seq, true_event = input_seq.to(device), true_event.to(device)
    with torch.no_grad():
        output = model(input_seq)
    total_loss = sum(
        F.cross_entropy(output[col], true_event[:, i], reduction='none').item()
        for i, col in enumerate(categorical_cols)
    )
    return total_loss"""

def compute_anomaly_score(model, input_seq, true_event, device):
    model.eval()
    input_seq, true_event = input_seq.to(device), true_event.to(device)
    with torch.no_grad():
        output = model(input_seq)

    for i, col in enumerate(categorical_cols):
        out = output[col]
        tgt = true_event[:, i]

        if tgt.item() >= out.size(1):
            print(f" ERRORE: col={col}, target={tgt.item()}, vocab_size={out.size(1)}")
            raise ValueError("Indice target fuori range per softmax")

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

def calculate_threshold(df, model, seq_len, device):
    scores = []

    # Step 1: Calcola tutti gli anomaly scores
    for idx in range(seq_len + 10, len(df), 10):
        sequence = df.iloc[idx - seq_len:idx]
        target = df.iloc[idx]

        x_seq = torch.tensor(sequence[categorical_cols].astype(int).values, dtype=torch.long).unsqueeze(0).to(device)
        y_target = torch.tensor(target[categorical_cols].astype(int).values, dtype=torch.long).unsqueeze(0).to(device)

        score = compute_anomaly_score(model, x_seq, y_target, device)
        scores.append(score)

    # Step 2: Calcola media e deviazione standard
    scores_np = np.array(scores)
    mean_score = scores_np.mean()
    std_score = scores_np.std()

    # Step 3: Calcola la soglia
    threshold = mean_score + std_score

    return threshold

def find_anomalies(df):
    return df[df['is_redteam'].astype(str) == '1'].index.tolist()

def evaluate_anomalies(df, model, anomaly_indices, seq_len, threshold, device):

    print(f"Totale righe marcate come anomalie (ground truth): {len(anomaly_indices)}")
    true_detected = 0

    for idx in anomaly_indices:
        try:
            # Calcolo del punteggio e determinazione dell'anomalia
            score, is_anom = is_event_anomalous(df, model, idx, seq_len, threshold, device)
            print(f"[{idx}] Score: {score:.2f} -> {'ANOMALO' if is_anom else 'normale'}")

            # Incremento del conteggio se l'anomalia è rilevata correttamente
            if is_anom:
                true_detected += 1
        except ValueError:
            # Gestione del caso in cui non è possibile calcolare l'anomalia
            continue

    # Calcolo della percentuale di anomalie rilevate correttamente
    detection_percentage = 100 * true_detected / len(anomaly_indices) if len(anomaly_indices) > 0 else 0
    print(f"Anomalie rilevate: {true_detected}")
    print(f"Percentuale rilevate correttamente: {detection_percentage:.2f}%")

    #return true_detected, detection_percentage