import torch.nn as nn

class TokenSpacingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, num_token_types=6, hidden_dim=128, output_dim=5, num_heads=4, num_layers=2):
        super(TokenSpacingModel, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.type_embedding = nn.Embedding(num_token_types, embedding_dim)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True),
            num_layers=num_layers
        )

        self.fc_type = nn.Linear(embedding_dim, output_dim)

    def forward(self, batch_input):
        tokens = batch_input[:, 0]
        types = batch_input[:, 1]

        token_embeddings = self.token_embedding(tokens)
        type_embeddings = self.type_embedding(types)

        combined_embeddings = token_embeddings + type_embeddings
        combined_embeddings = combined_embeddings.unsqueeze(1)

        transformer_output = self.transformer_encoder(combined_embeddings.permute(1, 0, 2))
        final_token_embedding = transformer_output[-1, :, :]

        type_pred = self.fc_type(final_token_embedding)
        return type_pred
