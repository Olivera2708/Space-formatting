import torch
import torch.nn as nn
import torch.nn.functional as F

class SpacingPredictionModel(nn.Module):
    def __init__(self, vocab_size, type_size, embedding_dim, hidden_dim):
        super(SpacingPredictionModel, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.type_embedding = nn.Embedding(type_size, embedding_dim)

        self.fc1 = nn.Linear(embedding_dim * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.spacing_type_fc = nn.Linear(hidden_dim, 4)
        self.spacing_length_fc = nn.Linear(hidden_dim, 1)

    def forward(self, input_batch):

        token1 = input_batch[0][0]
        type1 = input_batch[0][1]
        token2 = input_batch[1][0]
        type2= input_batch[1][1]

        # Get embeddings for tokens and types
        emb_token1 = self.token_embedding(token1)
        emb_type1 = self.type_embedding(type1)
        emb_token2 = self.token_embedding(token2)
        emb_type2 = self.type_embedding(type2)

        # Concatenate all embeddings for the pair
        combined_features = torch.cat((emb_token1, emb_type1, emb_token2, emb_type2), dim=1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(combined_features))
        x = F.relu(self.fc2(x))

        # Output spacing type (categorical) and spacing length (continuous)
        spacing_type_pred = self.spacing_type_fc(x)  # Outputs (3,) for 3 spacing types
        spacing_length_pred = self.spacing_length_fc(x)  # Outputs (1,) for spacing length

        return spacing_type_pred, spacing_length_pred

# Example usage:
# Define the model
model = SpacingPredictionModel(vocab_size=1000, type_size=6, embedding_dim=50, hidden_dim=128)

# Example input: batch of 1 with 1 pair (token1, type1, token2, type2)
input_batch = torch.tensor([[1, 0], [2,9]])  # (token1, type1, token2, type2)

# Forward pass
spacing_type_pred, spacing_length_pred = model(input_batch)

print("Spacing Type Prediction:", spacing_type_pred)
print("Spacing Length Prediction:", spacing_length_pred)
