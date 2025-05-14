import torch
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import seaborn as sns

from data_utils import AuthSequenceDataset, load_and_encode_data
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

# ======== Load & Split Train Data ========
df_full, encoders = load_and_encode_data(csv_train,1,100000)

# Split in 70% train, 30% validation
df_train, df_val = train_test_split(df_full, test_size=0.3, shuffle=False, random_state=42)

# Dataset & Dataloaders
train_dataset = AuthSequenceDataset(df_train, seq_len)
val_dataset = AuthSequenceDataset(df_val, seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 1 sample per time for scoring

# ======== Load Evaluation Set (Ground Truth) ========
df_eval, encoders_eval = load_and_encode_data(csv_test, 0,-1)
eval_dataset = AuthSequenceDataset(df_eval, seq_len)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# ======== Init Model ========
vocab_sizes = [len(encoders[col].classes_) for col in categorical_cols]
model = LSTMAnomalyModel(num_features=len(categorical_cols), vocab_sizes=vocab_sizes)
#model = BiLSTMAnomalyModel(num_features=len(categorical_cols), vocab_sizes=vocab_sizes)
#model = TransformerAnomalyModel(num_features=len(categorical_cols), vocab_sizes=vocab_sizes)


# ======== Train ========
losses = train_model(model, train_loader, device=device, epochs=epochs)
#plot_training_loss(losses)

# ======== Calculate Threshold on Validation Set ========
threshold  = calculate_threshold(df_val, model, seq_len, device=device)

print(threshold)
"""plt.figure(figsize=(10, 6))

# Istogramma + curva di densità
sns.histplot(scores, bins=30, kde=True, color='skyblue', edgecolor='black')

plt.title('Distribuzione degli Anomaly Scores')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequenza / Densità')
plt.grid(True)
plt.show()"""

# ======== Evaluate Anomalies on Evaluation Set (Ground Truth) ========
anomaly_indices = find_anomalies(df_eval)
cm1 = evaluate_anomalies(df_eval, model, anomaly_indices, seq_len, threshold, device)
compute_metrics(cm1)


print("test con pippo")
pippo = [i for i in range(0, len(df_val), 3)]
cm2 = evaluate_true_negatives(df_val, model, pippo, seq_len, threshold, device)
compute_metrics(cm2)

cm3 = cm1 + cm2
compute_metrics(cm3)



