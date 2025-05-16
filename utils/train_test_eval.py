import torch
import random
import numpy as np
from torch.ao.nn.quantized import Dropout
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import seaborn as sns

from data_utils import AuthSequenceDataset, load_and_encode_data, scrivi_riga
from lstm_model import *
from model_utils import *
from bilstm_model import *
from transformer_model import *


# ======== Config ========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

csv_train = "../data/processed/7withredSplit.csv"
csv_test = "../data/processed/8Test.csv"
seq_len = 10
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 8
Dropout = True
n_righe = 100000
#model = "LSTM"
#model = "BiLSTM"
model = "Transformer"

# ======== Load & Split Train Data ========
df_full, encoders = load_and_encode_data(csv_train,1,n_righe)

# Split in 70% train, 30% validation
df_train, df_val = train_test_split(df_full, test_size=0.3, shuffle=False, random_state=42)

# Dataset & Dataloaders
train_dataset = AuthSequenceDataset(df_train, seq_len)
val_dataset = AuthSequenceDataset(df_val, seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ======== Load Evaluation Set (Ground Truth) ========
df_test, encoders_eval = load_and_encode_data(csv_test, 0,-1)
eval_dataset = AuthSequenceDataset(df_test, seq_len)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# ======== Init Model ========
vocab_sizes = [len(encoders[col].classes_) for col in categorical_cols]
if model == "LSTM":
    model = LSTMAnomalyModel(num_features=len(categorical_cols), vocab_sizes=vocab_sizes, DROP=Dropout)
if model == "BiLSTM":
    model = BiLSTMAnomalyModel(num_features=len(categorical_cols), vocab_sizes=vocab_sizes, DROP=Dropout)
if model == "Transformer":
    model = TransformerAnomalyModel(num_features=len(categorical_cols), vocab_sizes=vocab_sizes, DROP=Dropout)


# ======== Train ========
losses = train_model(model, train_loader, device=device, epochs=epochs)
#plot_training_loss(losses)

# ======== Calculate Threshold on Validation Set ========
mean_score, max_score = calculate_threshold(df_val, model, seq_len, device=device)
threshold_range = np.arange(max_score, mean_score, -1)

valoreF1 = 0
valoreThreshold = 0

for threshold in threshold_range:
    print(f"\n--- Valutazione con threshold = {threshold:.2f} ---")

    # ======== Evaluate Anomalies on Evaluation Set (Ground Truth) ========
    anomaly_indices = find_anomalies(df_test)
    cm1 = evaluate_anomalies(df_test, model, anomaly_indices, seq_len, threshold, device)
    compute_metrics(cm1)

    # ======== Evaluate Regular on Validation Set ========
    pippo = [i for i in range(0, 2600, 10)]
    cm2 = evaluate_regular(df_val, model, pippo, seq_len, threshold, device)
    compute_metrics(cm2)

    # ======== Combined Metrics ========
    cm3 = cm1 + cm2
    metrics = compute_metrics(cm3)
    f1 = metrics["f1"]
    if f1 > valoreF1 :
        valoreF1 = f1
        valoreThreshold = threshold
        confusion_matrix = cm3

print("la threshold migliore è ", valoreThreshold , " con f1 pari a ", valoreF1)
print("la confusion matrix associata è : ")
met = compute_metrics(confusion_matrix,PRINT=True)

scrivi_riga(model,Dropout,epochs,n_righe,losses,met,valoreThreshold)







