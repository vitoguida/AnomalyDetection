import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.data_utils import categorical_cols  # List of column names for categorical features


class Attention(nn.Module):
    """
    Implements an attention mechanism that computes a weighted sum over LSTM outputs.
    This allows the model to focus on the most relevant time steps for prediction.
    """

    def __init__(self, hidden_dim):
        """
        Initializes the Attention module.

        Args:
            hidden_dim (int): Dimension of the hidden state output by the LSTM.
        """
        super(Attention, self).__init__()
        # Learnable attention weights of shape (hidden_dim,)
        self.attention_weights = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, lstm_outputs):
        """
        Applies attention mechanism to LSTM outputs.

        Args:
            lstm_outputs (Tensor): Output from the LSTM of shape (batch_size, seq_len, hidden_dim).

        Returns:
            context_vector (Tensor): Weighted sum of LSTM outputs of shape (batch_size, hidden_dim).
        """
        # Compute attention scores via dot product between LSTM outputs and attention weights
        attention_scores = torch.matmul(lstm_outputs, self.attention_weights)
        # Normalize attention scores using softmax along the sequence dimension
        attention_weights = F.softmax(attention_scores, dim=1)
        # Compute context vector as the weighted sum of LSTM outputs
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * lstm_outputs, dim=1)
        return context_vector


class BiLSTMAnomalyModel(nn.Module):
    """
    Bidirectional LSTM model with an attention mechanism for anomaly detection
    in time series data consisting of categorical variables.
    """

    def __init__(self, num_features, vocab_sizes, emb_dim=32, hidden_dim=64, lstm_dropout=0.3, DROP=False):
        """
        Initializes the BiLSTMAnomalyModel.

        Args:
            num_features (int): Number of categorical features.
            vocab_sizes (List[int]): Vocabulary sizes for each categorical feature.
            emb_dim (int): Dimension of the embedding vectors.
            hidden_dim (int): Dimension of the LSTM hidden state.
            lstm_dropout (float): Dropout rate between LSTM layers.
            DROP (bool): Whether to apply dropout after the attention mechanism.
        """
        super().__init__()

        # Create an embedding layer for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim) for vocab_size in vocab_sizes
        ])

        self.DROP = DROP
        self.hidden_dim = hidden_dim

        # Define a bidirectional LSTM; apply dropout only if specified
        if DROP:
            self.lstm = nn.LSTM(
                input_size=emb_dim * num_features,
                hidden_size=hidden_dim,
                batch_first=True,
                dropout=lstm_dropout,  # Dropout between LSTM layers
                num_layers=2,
                bidirectional=True  # Use bidirectional LSTM
            )
            self.dropout = nn.Dropout(p=lstm_dropout)
        else:
            self.lstm = nn.LSTM(
                input_size=emb_dim * num_features,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True
            )

        # Attention layer (hidden_dim * 2 because of bidirectional LSTM)
        self.attention = Attention(hidden_dim * 2)

        # Output layers: one linear layer per categorical column
        self.output_layers = nn.ModuleDict({
            col: nn.Linear(hidden_dim * 2, vocab_sizes[i])
            for i, col in enumerate(categorical_cols)
        })

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, num_features),
                        where each feature is an index into its vocabulary.

        Returns:
            Dict[str, Tensor]: A dictionary mapping each categorical column name to
                               its predicted logits (before softmax).
        """
        # Apply embeddings to each feature and concatenate them
        embedded = [self.embeddings[i](x[:, :, i]) for i in range(len(categorical_cols))]
        x_embed = torch.cat(embedded, dim=-1)  # Shape: (batch_size, seq_len, emb_dim * num_features)

        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x_embed)  # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)

        # Apply attention to get a single context vector per sequence
        context_vector = self.attention(lstm_out)  # Shape: (batch_size, hidden_dim * 2)

        # Optionally apply dropout
        if self.DROP:
            context_vector = self.dropout(context_vector)

        # Compute output logits for each categorical column
        return {
            col: self.output_layers[col](context_vector)  # Each output is (batch_size, vocab_size)
            for col in categorical_cols
        }
