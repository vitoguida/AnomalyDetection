# ========================== Imports ==========================
import random
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils.data_utils import *  # Utility functions for loading and processing data
from utils.model_utils import *  # Model training, evaluation, and plotting utilities
from models.lstm_model import *  # LSTM model implementation
from models.bilstm_model import *  # BiLSTM model implementation
from models.transformer_model import *  # Transformer model implementation


# ========================== Configuration ==========================
def set_seed(seed=42):
    """
    Sets random seed across various libraries for reproducibility.

    Args:
        seed (int): The random seed to use (default is 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-tuning to preserve determinism


# Set random seed
set_seed(42)

# File paths and parameters
csv_train = "data/processed/7withredSplit.csv"
csv_test = "data/processed/8Test.csv"
seq_len = 10  # Sequence length for model input
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
epochs = 10
Dropout = True  # Whether to apply dropout in the model
rows = 20000  # Number of rows to read from the training CSV
model = "LSTM"  # Choose model: "LSTM", "BiLSTM", or "Transformer"
#model = "BiLSTM"
#model = "Transformer"

# ========================== Load & Split Train Data ==========================
# Load and encode training data
df_full, encoders = load_and_encode_data(csv_train,1, rows)

# Split dataset into 70% training and 30% validation (no shuffle for time series integrity)
df_train, df_val = train_test_split(df_full, test_size=0.3, shuffle=False, random_state=42)

# Create dataset and dataloader for training
train_dataset = AuthSequenceDataset(df_train, seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# ========================== Load Evaluation Set ==========================
# Load and encode test data for anomaly evaluation
df_test, encoders_eval = load_and_encode_data(csv_test,0, -1) #the third parameter is for the number of rows , -1 means all the rows

# ========================== Model Initialization ==========================
# Determine vocab size for each categorical feature
vocab_sizes = [len(encoders[col].classes_) for col in categorical_cols]

# Initialize the chosen anomaly detection model
if model == "LSTM":
    model = LSTMAnomalyModel(num_features=len(categorical_cols), vocab_sizes=vocab_sizes, DROP=Dropout)
if model == "BiLSTM":
    model = BiLSTMAnomalyModel(num_features=len(categorical_cols), vocab_sizes=vocab_sizes, DROP=Dropout)
if model == "Transformer":
    model = TransformerAnomalyModel(num_features=len(categorical_cols), vocab_sizes=vocab_sizes, DROP=Dropout)


# ========================== Train Model ==========================
# Train the model using the training dataloader
losses = train_model(model, train_loader, device=device, epochs=epochs)
print(losses)

# Optional: plot training loss over epochs
# plot_training_loss(losses)

# ========================== Validation Threshold Selection ==========================
# Compute mean and max reconstruction error on validation set
mean_score, max_score = calculate_threshold(df_val, model, seq_len, device=device)

# Define a range of threshold values from max to mean (descending)
threshold_range = np.arange(max_score, mean_score, -1)

valoreF1 = 0
valoreThreshold = 0

# Randomly sample indices from validation set to evaluate "normal" behavior
pippo = random.sample(range(11, len(df_val)), 260)
tpr_list = []
fpr_list = []
thresholds = []

# Evaluate model performance across different threshold values
for threshold in threshold_range:
    print(f"\n--- Valutazione con threshold = {threshold:.2f} ---")

    # ======== Evaluate Anomalies on Evaluation Set (Ground Truth) ========
    anomaly_indices = find_anomalies(df_test)
    cm1 = evaluate_anomalies(df_test, model, anomaly_indices, seq_len, threshold, device)
    compute_metrics(cm1)

    # ======== Evaluate Regular on Validation Set ========
    cm2 = evaluate_regular(df_val, model, pippo, seq_len, threshold, device)
    compute_metrics(cm2)

    # ======== Combined Metrics ========
    cm3 = cm1 + cm2
    metrics = compute_metrics(cm3)
    # Calculate True Positive Rate and False Positive Rate for ROC curve
    tp = metrics["TP"]
    fn = metrics["FN"]
    fp = metrics["FP"]
    tn = metrics["TN"]

    tpr = tp / (tp + fn + 1e-6)
    fpr = fp / (fp + tn + 1e-6)

    tpr_list.append(tpr)
    fpr_list.append(fpr)
    thresholds.append(threshold)

    # Update best F1 score and corresponding threshold
    f1 = metrics["f1"]
    if f1 > valoreF1 :
        valoreF1 = f1
        valoreThreshold = threshold
        confusion_matrix = cm3


# ========================== Final Evaluation ==========================
# Print the best threshold and its associated F1 score
print("la threshold migliore è ", valoreThreshold, " con f1 pari a ", valoreF1)
print("la confusion matrix associata è : ")

# Compute and print detailed metrics for the best threshold
met = compute_metrics(confusion_matrix, PRINT=True)

# Plot ROC curve using collected TPR and FPR values
plot_roc(tpr_list, fpr_list)

# Log experiment results to file
write_results_tocsv(model, Dropout, epochs, rows, losses, met, valoreThreshold)







