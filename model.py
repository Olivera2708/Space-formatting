import torch

def get_batch(data, batch_size=32, block_size=256):
    random_ints = torch.randint(len(data)-block_size, (batch_size,))
    inputs = torch.stack([data[i:i+block_size] for i in random_ints])
    outputs = torch.stack([data[i+1:i+1+block_size] for i in random_ints])
    return inputs, outputs