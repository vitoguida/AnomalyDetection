import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import openpyxl
from openpyxl import load_workbook
from datetime import datetime


# Colonne categoriche da usare nel modello
categorical_cols = ["src_user", "src_domain", "dst_user", "dst_domain",
                    "src_comp", "dst_comp", "auth_type", "logon_type",
                    "orientation", "success"]

def load_and_encode_data(csv_path, start=2500000, end=2580000):
    df = pd.read_csv(csv_path, header=None, dtype=str, low_memory=False)
    if end == -1:
       end = df.shape[0]
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


"""plt.figure(figsize=(10, 6))

# Istogramma + curva di densità
sns.histplot(scores, bins=30, kde=True, color='skyblue', edgecolor='black')

plt.title('Distribuzione degli Anomaly Scores')
plt.xlabel('Anomaly Score')
plt.ylabel('Frequenza / Densità')
plt.grid(True)
plt.show()"""



excel_path = "risultati_modelli.xlsx"

def scrivi_riga(model,Dropout,epochs,n_righe,losses,metrics,valoreThreshold):
    nuova_riga = [
        str(model.__class__.__name__),  # Modello
        Dropout,  # Drop
        epochs,  # Epoch
        n_righe,  # RigheTrainTest
        round(losses[-1], 4),  # Loss (ultimo valore)
        metrics["TP"],  # TP
        metrics["FP"],  # FP
        metrics["TN"],  # TN
        metrics["FN"],  # FN
        round(valoreThreshold, 2),  # BestThreshold
        round(metrics["accuracy"], 4),  # Accuracy
        round(metrics["precision"], 4),  # Precision
        round(metrics["recall"], 4),  # Recall
        round(metrics["f1"], 4),  # f1-score
    ]

    # Aggiunta in fondo al file
    try:
        workbook = load_workbook(excel_path)
        sheet = workbook.active
    except FileNotFoundError:
        # Se il file non esiste, lo crea con intestazioni
        from openpyxl import Workbook
        workbook = Workbook()
        sheet = workbook.active
        sheet.append([
            "Modello", "Drop", "Epoch", "RigheTrainTest", "Loss",
            "TP", "FP", "TN", "FN", "BestThreshold",
            "Accuracy", "Precision", "Recall", "f1-score"
        ])

    sheet.append(nuova_riga)
    workbook.save(excel_path)

