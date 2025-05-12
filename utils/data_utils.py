import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

# Colonne categoriche da usare nel modello
categorical_cols = ["src_user", "src_domain", "dst_user", "dst_domain",
                    "src_comp", "dst_comp", "auth_type", "logon_type",
                    "orientation", "success"]

def load_and_encode_data(csv_path, start=2500000, end=2580000):
    df = pd.read_csv(csv_path, header=None, dtype=str, low_memory=False)
    df = df[start:end].reset_index(drop=True)
    df.columns = ["time", "src_user", "src_domain", "dst_user", "dst_domain",
                  "src_comp", "dst_comp", "auth_type", "logon_type", "orientation",
                  "success", "is_redteam"]

    encoders = {}
    for col in categorical_cols:
        df[col] = df[col].astype(str).fillna("UNK")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders

class AuthSequenceDataset(Dataset):
    def __init__(self, df, seq_len):
        self.seq_len = seq_len
        self.data = [
            (df.iloc[i:i+seq_len][categorical_cols].values,
             df.iloc[i+seq_len][categorical_cols].values)
            for i in range(len(df) - seq_len)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_seq, y_target = self.data[idx]
        x_seq = torch.tensor(x_seq, dtype=torch.long)
        y_target = torch.tensor(y_target.astype(int), dtype=torch.long)
        return x_seq, y_target
