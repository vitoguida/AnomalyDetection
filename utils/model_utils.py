import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

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
    max_score = scores_np.max()

    # Step 3: Calcola la soglia
    #threshold = mean_score + std_score


    return int(mean_score), int(max_score)

def find_anomalies(df):
    return df[df['is_redteam'].astype(str) == '1'].index.tolist()

def evaluate_anomalies(df, model, anomaly_indices, seq_len, threshold, device):
    print(f"Totale righe marcate come anomalie (ground truth): {len(anomaly_indices)}")
    true_detected = 0

    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for idx in anomaly_indices:
        try:
            # Calcolo del punteggio e determinazione dell'anomalia
            score, is_anom = is_event_anomalous(df, model, idx, seq_len, threshold, device)
            # print(f"[{idx}] Score: {score:.2f} -> {'ANOMALO' if is_anom else 'normale'}")

            # Incremento del conteggio se l'anomalia è rilevata correttamente
            if is_anom:
                true_detected += 1
                TP = TP + 1
            else : FN = FN + 1


        except ValueError:
            # Gestione del caso in cui non è possibile calcolare l'anomalia
            continue

    # Calcolo della percentuale di anomalie rilevate correttamente
    detection_percentage = 100 * true_detected / len(anomaly_indices) if len(anomaly_indices) > 0 else 0
    print(f"Anomalie rilevate: {true_detected}")
    print(f"Percentuale rilevate correttamente: {detection_percentage:.2f}%")

    confusion_matrix = [[TN, FP], [FN, TP]]

    return np.array(confusion_matrix)

def evaluate_regular(df, model, normal_indices, seq_len, threshold, device):

    print(f"Totale righe normali (ground truth): {len(normal_indices)}")
    true_negatives = 0
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for idx in normal_indices:
        try:
            # Calcolo del punteggio e determinazione dell'anomalia
            score, is_anom = is_event_anomalous(df, model, idx, seq_len, threshold, device)

            # Se il modello classifica correttamente come normale
            if not is_anom:
                true_negatives += 1
                TN = TN + 1
            else : FP = FP + 1
        except ValueError:
            continue

    # Calcolo della percentuale di veri negativi
    true_negative_rate = 100 * true_negatives / len(normal_indices) if len(normal_indices) > 0 else 0
    print(f"Veri negativi rilevati: {true_negatives}")
    print(f"Percentuale veri negativi (specificità): {true_negative_rate:.2f}%")
    confusion_matrix = [[TN, FP], [FN, TP]]
    return np.array(confusion_matrix)



def compute_metrics(cm,PRINT=False):
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * (precision * recall) / (precision + recall + 1e-6)
    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    if PRINT:
        print("\n📈 Metriche calcolate:")
        print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }