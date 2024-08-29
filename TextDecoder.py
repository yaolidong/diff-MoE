import torch
import torch.nn as nn

class TextDecoder(nn.Module):
    def __init__(self, input_dim, vocab_size, max_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.rnn = nn.GRU(input_dim, input_dim, batch_first=True)
        self.fc = nn.Linear(input_dim, vocab_size)
        self.max_length = max_length

    def forward(self, x, target=None):
        batch_size = x.size(0)
        if target is None:
            target = torch.zeros(batch_size, self.max_length, dtype=torch.long, device=x.device)
        
        embedded = self.embedding(target)
        output, _ = self.rnn(embedded, x.unsqueeze(0))
        logits = self.fc(output)
        
        # 确保输出的形状是 (batch_size, max_length, vocab_size)
        return logits.view(batch_size, self.max_length, -1)
