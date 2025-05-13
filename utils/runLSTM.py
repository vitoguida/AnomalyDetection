import torch
import random
import numpy as np
from torch.utils.data import DataLoader

from data_utils import AuthSequenceDataset,load_and_encode_data
from lstm_model import *
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


csv_train = "../data/processed/7withredSplit.csv"
#csv_test = "../data/processed/8Test.csv"
csv_test = "../data/processed/8withredSplit.csv"
seq_len = 10
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ======== Load Data ========
#train dataset
df, encoders = load_and_encode_data(csv_train, start=1, end=100000)
dataset = AuthSequenceDataset(df, seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#test dataset
dfTest, encodersTest = load_and_encode_data(csv_test, start=1, end=3000)
datasetTest = AuthSequenceDataset(dfTest, seq_len)
dataloaderTest = DataLoader(datasetTest, batch_size=batch_size, shuffle=False)

# ======== Init Model ========
vocab_sizes = [len(encoders[col].classes_) for col in categorical_cols]
model = LSTMAnomalyModel(num_features=len(categorical_cols), vocab_sizes=vocab_sizes)

# ======== Train ========
losses = train_model(model, dataloader, device=device, epochs=10)
plot_training_loss(losses)

# ======== Evaluate Anomalies ========
threshold = 10.0
#anomaly_indices = find_anomalies(dfTest)
anomaly_indices = [23,45,67,89,90,14,1234,65,234,567,876,909]



evaluate_anomalies(dfTest, model, anomaly_indices, seq_len, threshold, device)

