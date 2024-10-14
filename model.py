import torch
import torch.nn as nn
import torch.nn.functional as F

class CodeFormatterModel(nn.Module):
    def __init__(self, vocab_size_word, vocab_size_type=6, embed_size=256, hidden_size=512):
        super(CodeFormatterModel, self).__init__()
        
        self.word_embedding = nn.Embedding(vocab_size_word, embed_size)
        self.type_embedding = nn.Embedding(vocab_size_type, embed_size)
        
        self.fc1 = nn.Linear(embed_size * 4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, input_batch):
        word1_idx, type1_idx, word2_idx, type2_idx = input_batch[:, 0], input_batch[:, 1], input_batch[:, 2], input_batch[:, 3]

        word1_embeds = self.word_embedding(word1_idx)
        type1_embeds = self.type_embedding(type1_idx)
        word2_embeds = self.word_embedding(word2_idx)
        type2_embeds = self.type_embedding(type2_idx)

        combined_embeds = torch.cat((word1_embeds, type1_embeds, word2_embeds, type2_embeds), dim=-1)

        combined_flat = combined_embeds.view(combined_embeds.size(0), -1)

        x = torch.relu(self.fc1(combined_flat))
        output = self.fc2(x)
        spacing_type = output[:, 0]
        length_of_spacing = output[:, 1]

        return spacing_type, length_of_spacing