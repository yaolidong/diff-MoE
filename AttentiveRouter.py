import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class AttentiveRouter(nn.Module):
    def __init__(self, hidden_size: int, top_k: int, num_experts: int):
        super().__init__()
        # 确保top_k不超过专家数量
        self.top_k = min(top_k, num_experts)
        self.num_experts = num_experts
        
        # 注意：如果top_k大于num_experts，发出警告
        if top_k > num_experts:
            print(f"警告: top_k ({top_k}) 大于专家数量 ({num_experts})，已自动调整为 {num_experts}")
        
        # 路由器网络
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, num_experts)
        )
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        for module in self.router.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 计算专家权重
        expert_weights = self.router(x)  # [batch_size, seq_len, num_experts]
        
        # 获取top-k专家
        top_k_weights, top_k_indices = torch.topk(expert_weights, self.top_k, dim=-1)
        
        # 计算softmax
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        
        # 创建mask
        masks = torch.zeros_like(expert_weights).scatter_(-1, top_k_indices, top_k_weights)
        
        # 计算路由损失
        # 1. 负载均衡损失
        expert_usage = masks.sum(dim=[0, 1])  # [num_experts]
        expert_usage = expert_usage / expert_usage.sum()  # 归一化
        target_usage = torch.ones_like(expert_usage) / self.num_experts
        load_balancing_loss = F.mse_loss(expert_usage, target_usage)
        
        # 2. 稀疏性损失
        sparsity_loss = torch.mean(torch.sum(masks > 0, dim=-1).float()) / self.top_k
        
        # 总损失
        total_loss = load_balancing_loss + 0.1 * sparsity_loss
        
        return {
            'weights': expert_weights,
            'masks': masks,
            'loss': total_loss,
            'expert_usage': expert_usage
        }
    
    def get_attention_weights(self):
        """获取最近一次forward的注意力权重"""
        return self.attention_weights 