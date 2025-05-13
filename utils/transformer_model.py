import torch
import torch.nn as nn 
from data_utils import categorical_cols

class TransformerAnomalyModel(nn.Module):
    def __init__(self, num_features, vocab_sizes, emb_dim=32, hidden_dim=64, nhead=4,
                 num_layers=2, transformer_dropout=0.3, DROP=False):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim) for vocab_size in vocab_sizes
        ])
        self.DROP = DROP
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.seq_feature_dim = emb_dim * num_features

        self.bn = nn.BatchNorm1d(self.seq_feature_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.seq_feature_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=transformer_dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if DROP:
            self.dropout = nn.Dropout(p=transformer_dropout)

        self.output_layers = nn.ModuleDict({
            col: nn.Linear(self.seq_feature_dim, vocab_sizes[i])
            for i, col in enumerate(categorical_cols)
        })

    def forward(self, x):
        embedded = [self.embeddings[i](x[:, :, i]) for i in range(len(categorical_cols))]
        x_embed = torch.cat(embedded, dim=-1)  # Shape: [B, T, F]

        # Reshape for batch norm: BN expects (B, C), so we flatten time and features
        B, T, F = x_embed.shape
        x_bn = self.bn(x_embed.view(B * T, F)).view(B, T, F)

        # Transformer expects input of shape [B, T, F]
        transformer_out = self.transformer_encoder(x_bn)

        # Get representation from the last token (or mean)
        sequence_rep = transformer_out[:, -1, :]  # Shape: [B, F]
        if self.DROP:
            sequence_rep = self.dropout(sequence_rep)

        return {col: self.output_layers[col](sequence_rep) for col in categorical_cols}
