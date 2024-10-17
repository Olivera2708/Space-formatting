import torch
import torch.nn as nn
import torch.optim as optim

class TokenSpacingModel(nn.Module):
    def __init__(self, input_dim, model_dim=128, output_dim=4, num_heads=8, num_layers=3, dropout=0.1):
        super(TokenSpacingModel, self).__init__()
        self.embedding = nn.Embedding(input_dim+1, model_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.fc = nn.Linear(model_dim, output_dim)
        
    def forward(self, x):
        padding_mask = (x != -1)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, src_key_padding_mask=~padding_mask)

        output_pairs = []
        for i in range(1, x.size(0) - 1):  # Start from 1 and go to n-1 to avoid out of bounds
            # Concatenate previous, current, and next token representations
            combined = torch.cat((x[i-1], x[i], x[i+1]), dim=-1)  # (model_dim * 3,)

        # Stack the outputs and return
        output_pairs = self.fc(combined)  # Shape: (sequence_length - 2, batch_size, output_dim)
        print(output_pairs)
        return output_pairs

