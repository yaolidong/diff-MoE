import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=10, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.output_dim = output_dim
        
        # 门控网络
        self.gate = nn.Linear(input_dim, num_experts)
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim)
            ) for _ in range(num_experts)
        ])
        
        # CLS token和向量表征
        self.cls = nn.Linear(output_dim, output_dim)
        self.vector = nn.Linear(output_dim, output_dim)
    
    def forward(self, x):
        # 确保输入是浮点型
        x = x.float()
        
        # 计算门控权重
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # 选择Top-k专家
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # 重新归一化
        
        # 应用选中的专家
        expert_outputs = torch.zeros(x.size(0), self.output_dim, device=x.device)
        for i, expert in enumerate(self.experts):
            mask = top_k_indices == i
            if mask.any():
                expert_inputs = x[mask.any(dim=1)]
                expert_output = expert(expert_inputs)
                expert_outputs[mask.any(dim=1)] += expert_output * top_k_probs[mask][:, None]
        
        # 生成CLS token和向量表征
        cls_output = self.cls(expert_outputs)
        vector_output = self.vector(expert_outputs)
        
        return cls_output, vector_output
