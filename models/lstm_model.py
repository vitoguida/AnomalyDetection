import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.data_utils import categorical_cols  # List of categorical column names used for output layer construction


class LSTMAnomalyModel(nn.Module):
    """
    LSTM-based anomaly detection model for categorical time series data.
    Each categorical feature is embedded, passed through an LSTM, and decoded into
    its original category space.
    """

    def __init__(self, num_features, vocab_sizes, emb_dim=32, hidden_dim=64, lstm_dropout=0.3, DROP=False):
        """
        Initializes the LSTMAnomalyModel.

        Args:
            num_features (int): Number of categorical features.
            vocab_sizes (List[int]): List of vocabulary sizes for each categorical feature.
            emb_dim (int): Dimensionality of the embedding vectors.
            hidden_dim (int): Number of hidden units in the LSTM.
            lstm_dropout (float): Dropout rate to use within the LSTM (only if DROP=True).
            DROP (bool): Whether to apply dropout in LSTM and before output layers.
        """
        super().__init__()

        # Create an embedding layer for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim) for vocab_size in vocab_sizes
        ])

        self.DROP = DROP

        if DROP:
            # LSTM with dropout applied between layers (requires num_layers ≥ 2)
            self.lstm = nn.LSTM(
                input_size=emb_dim * num_features,  # Total input size after concatenating all embeddings
                hidden_size=hidden_dim,
                batch_first=True,
                dropout=lstm_dropout,  # Dropout between LSTM layers
                num_layers=2
            )

            # Explicit dropout layer after LSTM output
            self.dropout = nn.Dropout(p=lstm_dropout)

            # Output layer: a separate linear classifier for each categorical feature
            self.output_layers = nn.ModuleDict({
                col: nn.Linear(hidden_dim, vocab_sizes[i])  # Predict logits for each category in col
                for i, col in enumerate(categorical_cols)
            })

        else:
            # LSTM without dropout (single-layer by default)
            self.lstm = nn.LSTM(
                input_size=emb_dim * num_features,
                hidden_size=hidden_dim,
                batch_first=True
            )

            # Output layers remain the same
            self.output_layers = nn.ModuleDict({
                col: nn.Linear(hidden_dim, vocab_sizes[i])
                for i, col in enumerate(categorical_cols)
            })

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, num_features),
                        where each feature is a categorical index.

        Returns:
            Dict[str, Tensor]: Dictionary mapping each categorical column name to its predicted logits.
        """
        # Apply embedding to each categorical feature across the time dimension
        embedded = [self.embeddings[i](x[:, :, i]) for i in range(len(categorical_cols))]

        # Concatenate all embedded features along the last dimension
        x_embed = torch.cat(embedded, dim=-1)  # Shape: (batch_size, sequence_length, emb_dim * num_features)

        # Pass the embedded sequence through the LSTM
        _, (h_n, _) = self.lstm(x_embed)  # h_n: (num_layers, batch_size, hidden_dim)

        # Extract the final hidden state from the last LSTM layer
        h = h_n[-1]  # Shape: (batch_size, hidden_dim)

        # Apply dropout if enabled
        if self.DROP:
            h = self.dropout(h)

        # Compute logits for each categorical feature using the final hidden state
        return {
            col: self.output_layers[col](h) for col in categorical_cols
        }
