import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentiveRouter(nn.Module):
    def __init__(self, embed_dim, num_experts, top_k=2, expert_capacity_factor=1.25):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity_factor = expert_capacity_factor
        
        # 注意力机制用于路由决策
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_experts)
        )
        
        # 用于计算专家分配的温度参数
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        # 用于记录注意力权重
        self.register_buffer('attention_weights', None, persistent=False)
        
    def forward(self, x):
        # 计算注意力分数
        attention_scores = self.attention(x)  # [batch_size, seq_len, num_experts]
        
        # 应用温度缩放
        attention_scores = attention_scores / self.temperature
        
        # 使用softmax获取专家分配概率
        expert_weights = F.softmax(attention_scores, dim=-1)
        
        # 保存注意力权重用于可视化
        self.attention_weights = expert_weights.detach()
        
        # 选择top-k个专家
        top_k_weights, top_k_indices = torch.topk(expert_weights, self.top_k, dim=-1)
        
        # 重新归一化top-k权重
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # 创建专家mask
        expert_mask = torch.zeros_like(expert_weights)
        expert_mask.scatter_(-1, top_k_indices, top_k_weights)
        
        # 计算路由损失（用于平衡专家使用）
        # 1. 负载平衡损失
        expert_usage = expert_mask.sum(dim=[0, 1])  # [num_experts]
        ideal_usage = expert_mask.sum() / self.num_experts
        load_balance_loss = torch.mean((expert_usage - ideal_usage).pow(2))
        
        # 2. 专家容量损失
        capacity = int(self.expert_capacity_factor * expert_mask.size(1))
        expert_capacity_loss = torch.relu(expert_usage - capacity).mean()
        
        # 总路由损失
        router_loss = load_balance_loss + expert_capacity_loss
        
        return {
            'expert_masks': expert_mask,
            'router_loss': router_loss,
            'attention_weights': self.attention_weights,
            'expert_indices': top_k_indices
        }
    
    def get_attention_weights(self):
        """获取最近一次forward的注意力权重"""
        return self.attention_weights 