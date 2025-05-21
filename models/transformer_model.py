import torch
import torch.nn as nn
from utils.data_utils import categorical_cols  # List of categorical column names used for output predictions


class TransformerAnomalyModel(nn.Module):
    """
    Transformer-based neural network for anomaly detection on sequential categorical data.

    This model embeds categorical features, applies a transformer encoder, and outputs
    class predictions for each categorical column.

    Args:
        num_features (int): Number of categorical features in the input.
        vocab_sizes (list[int]): Vocabulary size for each categorical feature.
        emb_dim (int): Dimension of embedding for each categorical feature.
        hidden_dim (int): Hidden layer size in the transformer feedforward network.
        nhead (int): Number of attention heads in the transformer.
        num_layers (int): Number of transformer encoder layers.
        transformer_dropout (float): Dropout rate used in the transformer layers.
        DROP (bool): If True, apply dropout before the final output.
    """

    def __init__(self, num_features, vocab_sizes, emb_dim=32, hidden_dim=64, nhead=4,
                 num_layers=2, transformer_dropout=0.3, DROP=False):
        super().__init__()

        # Embedding layers for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim) for vocab_size in vocab_sizes
        ])

        self.DROP = DROP
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.seq_feature_dim = emb_dim * num_features  # Total feature size after concatenation of all embeddings

        # Batch normalization for stabilizing learning
        self.bn = nn.BatchNorm1d(self.seq_feature_dim)

        # Define a single transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.seq_feature_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=transformer_dropout,
            batch_first=True  # Use (B, T, F) input format
        )

        # Stack multiple transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Optional dropout before the final output layer
        if DROP:
            self.dropout = nn.Dropout(p=transformer_dropout)

        # Output linear layers for each categorical column
        # Each outputs logits of shape [B, vocab_size] for its respective feature
        self.output_layers = nn.ModuleDict({
            col: nn.Linear(self.seq_feature_dim, vocab_sizes[i])
            for i, col in enumerate(categorical_cols)
        })

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape [B, T, F], where
                        B = batch size, T = sequence length, F = number of features.

        Returns:
            dict[str, Tensor]: A dictionary mapping each categorical column name
                               to its predicted logits tensor of shape [B, vocab_size].
        """
        # Apply embedding to each feature across all time steps
        embedded = [self.embeddings[i](x[:, :, i]) for i in range(len(categorical_cols))]

        # Concatenate embeddings along the last dimension -> shape: [B, T, F * emb_dim]
        x_embed = torch.cat(embedded, dim=-1)

        # Reshape for batch normalization: flatten batch and time dims to apply BN on features
        B, T, F = x_embed.shape
        x_bn = self.bn(x_embed.view(B * T, F)).view(B, T, F)

        # Pass normalized embeddings through transformer encoder
        transformer_out = self.transformer_encoder(x_bn)  # Output shape: [B, T, F]

        # Extract the representation from the last time step token
        sequence_rep = transformer_out[:, -1, :]  # Shape: [B, F]

        # Optionally apply dropout for regularization
        if self.DROP:
            sequence_rep = self.dropout(sequence_rep)

        # Generate output logits for each categorical column using its respective output layer
        return {col: self.output_layers[col](sequence_rep) for col in categorical_cols}
