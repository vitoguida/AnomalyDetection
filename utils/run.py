import torch
import random
import numpy as np
from torch.utils.data import DataLoader

from data_utils import AuthSequenceDataset,load_and_encode_data
from lstm_model import *
from bilstm_model import *
from model_utils import *



# ======== Config ========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

csv_path = "../data/processed/8withredSplit.csv"
seq_len = 10
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ======== Load Data ========
df, encoders = load_and_encode_data(csv_path)
dataset = AuthSequenceDataset(df, seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ======== Init Model ========
vocab_sizes = [len(encoders[col].classes_) for col in categorical_cols]
#model = LSTMAnomalyModel(num_features=len(categorical_cols), vocab_sizes=vocab_sizes)
modelBilstm = BiLSTMAnomalyModel(num_features=len(categorical_cols), vocab_sizes=vocab_sizes)

# ======== Train ========
#losses = train_model(model, dataloader, device=device, epochs=30)
#plot_training_loss(losses)

lossesBilstm = train_model(modelBilstm, dataloader, device=device, epochs=30)
plot_training_loss(lossesBilstm)

# ======== Evaluate Anomalies ========
threshold = 10.0
anomaly_indices = find_anomalies(df)


evaluate_anomalies(df, modelBilstm, anomaly_indices, seq_len, threshold, device)

