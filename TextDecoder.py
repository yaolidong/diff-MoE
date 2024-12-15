import torch
import torch.nn as nn

class TextDecoder(nn.Module):
    def __init__(self, vocab_size, max_length=16, embed_dim=1024, hidden_dim=512):
        super().__init__()
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # LSTM解码器
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=False
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, vocab_size)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 投影特征
        hidden = self.projection(x)
        
        # 检查维度并正确扩展为序列
        if len(hidden.shape) == 2:  # [batch_size, hidden_dim]
            hidden = hidden.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            hidden = hidden.expand(-1, self.max_length, -1)  # [batch_size, max_length, hidden_dim]
        elif len(hidden.shape) == 3:  # 如果已经是3维，确保序列长度正确
            hidden = hidden.expand(-1, self.max_length, -1)
        
        # LSTM解码
        lstm_out, _ = self.lstm(hidden)
        
        # 生成词表大小的logits
        logits = self.output_layer(lstm_out)
        
        return logits