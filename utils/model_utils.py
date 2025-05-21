import torch
import torch.nn.functional as F
from .data_utils import categorical_cols
import numpy as np

def train_model(model, dataloader, device, epochs=10):
    """
    Trains the model using cross-entropy loss on categorical output columns.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        dataloader (DataLoader): Batches of input and target sequences.
        device (torch.device): CPU or GPU device.
        epochs (int): Number of training epochs.

    Returns:
        List[float]: Average loss per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Zero gradients before backward pass
            optimizer.zero_grad()

            # Forward pass
            outputs = model(x_batch)

            # Compute total cross-entropy loss for each categorical column
            loss = sum(F.cross_entropy(outputs[col], y_batch[:, i])
                       for i, col in enumerate(categorical_cols))

            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return epoch_losses

def plot_training_loss(losses):
    """
    Plots the training loss over epochs.

    Args:
        losses (List[float]): Average loss values per epoch.
    """
    import matplotlib.pyplot as plt
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

def compute_anomaly_score(model, input_seq, true_event, device):
    """
    Computes the anomaly score (sum of cross-entropy losses) for a single input-target pair.

    Args:
        model (torch.nn.Module): Trained model.
        input_seq (Tensor): Input sequence tensor of shape (1, seq_len, features).
        true_event (Tensor): Target event tensor of shape (1, num_features).
        device (torch.device): CPU or GPU device.

    Returns:
        float: Anomaly score (higher indicates more anomalous).
    """
    model.eval()
    input_seq, true_event = input_seq.to(device), true_event.to(device)

    with torch.no_grad():
        output = model(input_seq)

    # Validate that the target indices do not exceed the vocabulary size
    for i, col in enumerate(categorical_cols):
        out = output[col]
        tgt = true_event[:, i]
        if tgt.item() >= out.size(1):
            print(f"ERROR: col={col}, target={tgt.item()}, vocab_size={out.size(1)}")
            raise ValueError("Target index out of range for softmax output")

    # Sum of individual cross-entropy losses (one per categorical column)
    total_loss = sum(
        F.cross_entropy(output[col], true_event[:, i], reduction='none').item()
        for i, col in enumerate(categorical_cols)
    )
    return total_loss

def is_event_anomalous(df, model, index, seq_len, threshold, device):
    """
    Determines if an event at a given index is anomalous based on a threshold.

    Args:
        df (DataFrame): Full dataset.
        model (torch.nn.Module): Trained model.
        index (int): Index of the event to check.
        seq_len (int): Length of the input sequence.
        threshold (float): Anomaly threshold.
        device (torch.device): CPU or GPU.

    Returns:
        Tuple[float, bool]: Anomaly score and True if the event is anomalous.
    """
    if index < seq_len:
        raise ValueError("Index too small to generate input sequence.")

    sequence = df.iloc[index - seq_len:index]
    target = df.iloc[index]

    x_seq = torch.tensor(sequence[categorical_cols].astype(int).values, dtype=torch.long).unsqueeze(0)
    y_target = torch.tensor(target[categorical_cols].astype(int).values, dtype=torch.long).unsqueeze(0)

    score = compute_anomaly_score(model, x_seq, y_target, device)
    return score, score > threshold

def calculate_threshold(df, model, seq_len, device):
    """
    Calculates the mean and max anomaly scores from the dataset to be used as thresholds.

    Args:
        df (DataFrame): Dataset.
        model (torch.nn.Module): Trained model.
        seq_len (int): Length of input sequences.
        device (torch.device): CPU or GPU.

    Returns:
        Tuple[int, int]: Mean and maximum anomaly scores.
    """
    scores = []

    # Sample data points every 10 steps to compute representative scores
    for idx in range(seq_len + 10, len(df), 10):
        sequence = df.iloc[idx - seq_len:idx]
        target = df.iloc[idx]

        x_seq = torch.tensor(sequence[categorical_cols].astype(int).values, dtype=torch.long).unsqueeze(0).to(device)
        y_target = torch.tensor(target[categorical_cols].astype(int).values, dtype=torch.long).unsqueeze(0).to(device)

        score = compute_anomaly_score(model, x_seq, y_target, device)
        scores.append(score)

    scores_np = np.array(scores)
    mean_score = scores_np.mean()
    std_score = scores_np.std()
    max_score = scores_np.max()

    # Optionally: return mean + std as threshold (commented out)
    # threshold = mean_score + std_score

    return int(mean_score), int(max_score)

def find_anomalies(df):
    """
    Identifies indices of rows labeled as anomalous (e.g., red team activity).

    Args:
        df (DataFrame): Dataset with 'is_redteam' label.

    Returns:
        List[int]: Indices of anomalous events.
    """
    return df[df['is_redteam'].astype(str) == '1'].index.tolist()

def evaluate_anomalies(df, model, anomaly_indices, seq_len, threshold, device):
    """
    Evaluates how many labeled anomalies are correctly detected by the model.

    Args:
        df (DataFrame): Dataset.
        model (torch.nn.Module): Trained model.
        anomaly_indices (List[int]): Ground truth anomalous indices.
        seq_len (int): Input sequence length.
        threshold (float): Anomaly threshold.
        device (torch.device): CPU or GPU.

    Returns:
        ndarray: Confusion matrix [[TN, FP], [FN, TP]].
    """
    print(f"Total ground truth anomalies: {len(anomaly_indices)}")
    true_detected = 0

    TP = FP = FN = TN = 0

    for idx in anomaly_indices:
        try:
            score, is_anom = is_event_anomalous(df, model, idx, seq_len, threshold, device)
            if is_anom:
                true_detected += 1
                TP += 1
            else:
                FN += 1
        except ValueError:
            continue

    detection_percentage = 100 * true_detected / len(anomaly_indices) if anomaly_indices else 0
    print(f"Anomalies detected: {true_detected}")
    print(f"Detection rate: {detection_percentage:.2f}%")

    return np.array([[TN, FP], [FN, TP]])

def evaluate_regular(df, model, normal_indices, seq_len, threshold, device):
    """
    Evaluates how many normal events are correctly identified as non-anomalous.

    Args:
        df (DataFrame): Dataset.
        model (torch.nn.Module): Trained model.
        normal_indices (List[int]): Ground truth normal indices.
        seq_len (int): Input sequence length.
        threshold (float): Anomaly threshold.
        device (torch.device): CPU or GPU.

    Returns:
        ndarray: Confusion matrix [[TN, FP], [FN, TP]].
    """
    print(f"Total ground truth normal events: {len(normal_indices)}")
    true_negatives = 0

    TP = FP = FN = TN = 0

    for idx in normal_indices:
        try:
            score, is_anom = is_event_anomalous(df, model, idx, seq_len, threshold, device)
            if not is_anom:
                true_negatives += 1
                TN += 1
            else:
                FP += 1
        except ValueError:
            continue

    true_negative_rate = 100 * true_negatives / len(normal_indices) if normal_indices else 0
    print(f"True negatives detected: {true_negatives}")
    print(f"Specificity: {true_negative_rate:.2f}%")

    return np.array([[TN, FP], [FN, TP]])

def compute_metrics(cm, PRINT=False):
    """
    Computes evaluation metrics from a confusion matrix.

    Args:
        cm (ndarray): Confusion matrix [[TN, FP], [FN, TP]].
        PRINT (bool): If True, prints the metrics.

    Returns:
        dict: Dictionary containing precision, recall, F1, accuracy, and counts.
    """
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * (precision * recall) / (precision + recall + 1e-6)
    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    if PRINT:
        print("\n📈 Evaluation Metrics:")
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
