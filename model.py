import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenSpacingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, num_token_types=6, hidden_dim=128, output_dim=4):
        super(TokenSpacingModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.type_embedding = nn.Embedding(num_token_types, embedding_dim)
        
        self.fc1 = nn.Linear(2 * embedding_dim, hidden_dim)
        self.fc_type = nn.Linear(hidden_dim, output_dim)
        self.fc_length = nn.Linear(hidden_dim, 1)
    
    def forward(self, batch_input):
        tokens = batch_input[:, 0]
        types = batch_input[:, 1]
        
        token_pairs = torch.stack((tokens[:-1], tokens[1:]), dim=1)
        type_pairs = torch.stack((types[:-1], types[1:]), dim=1)
        
        token1_embedding = self.token_embedding(token_pairs[:, 0])
        token2_embedding = self.token_embedding(token_pairs[:, 1])
        
        type1_embedding = self.type_embedding(type_pairs[:, 0])
        type2_embedding = self.type_embedding(type_pairs[:, 1])
        
        combined_token_embedding = token1_embedding + token2_embedding
        combined_type_embedding = type1_embedding + type2_embedding
        
        combined_embedding = torch.cat((combined_token_embedding, combined_type_embedding), dim=1)
        
        x = torch.relu(self.fc1(combined_embedding))
        
        type_pred = self.fc_type(x)
        length_pred = self.fc_length(x)
        
        return type_pred, length_pred
