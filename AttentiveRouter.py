import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import math

class AttentiveRouter(nn.Module):
    def __init__(self, hidden_size: int, top_k: int, num_experts: int):
        super().__init__()
        # 确保top_k不超过专家数量
        self.top_k = min(top_k, num_experts)
        self.num_experts = num_experts
        
        # 注意：如果top_k大于num_experts，发出警告
        if top_k > num_experts:
            print(f"警告: top_k ({top_k}) 大于专家数量 ({num_experts})，已自动调整为 {num_experts}")
        
        # 使用更轻量级的路由器网络，但增加非线性能力
        self.router = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_experts)
        )
        
        # 初始化参数
        self._init_weights()
        
        # 调整稳定性参数
        self.eps = 1e-6
        self.temperature = 0.7  # 降低温度参数以获得更锐利的分布
        self.router_capacity_factor = 2.0  # 增加专家容量因子
        
        # 添加专家容量跟踪
        register_buffer = getattr(self, "register_buffer", None)
        if callable(register_buffer):
            self.register_buffer("_expert_count", torch.zeros(num_experts))
        else:
            self._expert_count = torch.zeros(num_experts)
            
    def _init_weights(self):
        """初始化路由器权重，采用更好的初始化方法"""
        for name, module in self.router.named_modules():
            if isinstance(module, nn.Linear):
                # 使用fan_in模式的kaiming初始化
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 获取输入形状
        batch_size, seq_len, hidden_size = x.shape
        
        # 计算路由器权重时添加额外的归一化步骤
        expert_weights = self.router(x)
        
        # 应用温度参数
        expert_weights = expert_weights / self.temperature
        
        # 数值稳定性处理
        expert_weights = torch.clamp(expert_weights, min=-50.0, max=50.0)
        
        # 计算专家容量
        capacity = int(self.router_capacity_factor * batch_size * seq_len * self.top_k / self.num_experts)
        
        # 重置专家计数
        if hasattr(self, "_expert_count"):
            self._expert_count.zero_()
        else:
            self._expert_count = torch.zeros(self.num_experts, device=expert_weights.device)
        
        # 获取top-k专家
        original_shape = expert_weights.shape
        expert_weights_flat = expert_weights.reshape(-1, self.num_experts)
        
        # 获取topk权重和索引
        top_k_weights, top_k_indices = torch.topk(expert_weights_flat, self.top_k, dim=-1)
        
        # 使用向量化操作创建mask
        masks_flat = torch.zeros_like(expert_weights_flat)
        top_k_weights_softmax = F.softmax(top_k_weights, dim=-1)
        
        # 使用索引高级操作代替循环
        batch_indices = torch.arange(masks_flat.size(0), device=masks_flat.device).unsqueeze(1).expand(-1, self.top_k)
        batch_indices = batch_indices.reshape(-1)
        expert_indices = top_k_indices.reshape(-1)
        values = top_k_weights_softmax.reshape(-1)
        
        # 使用scatter_替代手动循环
        masks_flat = masks_flat.clone()
        masks_flat[batch_indices, expert_indices] = values
        
        # 实现专家容量限制
        for idx in range(self.num_experts):
            # 获取当前专家的权重
            expert_mask = masks_flat[:, idx]
            expert_sum = expert_mask.sum().item()
            
            # 如果专家分配超过容量
            if expert_sum > capacity:
                # 使用topk一次性找出最重要的tokens
                expert_values, expert_indices = torch.topk(expert_mask, min(capacity, (expert_mask > 0).sum().item()))
                
                # 高效创建新掩码
                new_mask = torch.zeros_like(expert_mask)
                new_mask[expert_indices] = expert_mask[expert_indices]
                
                # 更新掩码
                masks_flat[:, idx] = new_mask
                self._expert_count[idx] = new_mask.sum().item()
            else:
                self._expert_count[idx] = expert_sum
                
        # 重新normalize每个token的mask，确保和为1
        mask_sum = masks_flat.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        masks_flat = masks_flat / mask_sum
        
        # 恢复原始形状
        masks = masks_flat.reshape(original_shape)
        
        # 负载均衡损失计算 - 使用改进的计算方式
        # 计算每个专家的使用频率
        expert_usage = self._expert_count / self._expert_count.sum().clamp(min=self.eps)
        
        # 计算目标使用频率
        target_usage = torch.ones_like(expert_usage) / self.num_experts
        
        # 计算KL散度作为负载均衡损失
        load_balancing_loss = F.kl_div(
            torch.log(expert_usage.clamp(min=self.eps)), 
            target_usage, 
            reduction='batchmean'
        )
        
        # 增加路由损失权重
        total_loss = 0.01 * load_balancing_loss
        
        return {
            'weights': expert_weights,
            'masks': masks,
            'loss': total_loss,
            'expert_usage': expert_usage
        }
    
    def get_attention_weights(self):
        """获取最近一次forward的注意力权重"""
        return self.attention_weights 