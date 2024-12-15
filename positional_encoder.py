import torch

def positional_encoding(batch_size, seq_length, d_model):
    position = torch.arange(0, seq_length).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
    
    pe = torch.zeros(seq_length, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)
    return pe 