import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentiveRouter(nn.Module):
    def __init__(self, d_model, top_k, num_experts, router_z_loss_coef=0.001, router_aux_loss_coef=0.001):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.router_z_loss_coef = router_z_loss_coef
        self.router_aux_loss_coef = router_aux_loss_coef
        
        # 注意力查询和键
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Parameter(torch.randn(num_experts, d_model))
        
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_experts)
        )
        
        # 缩放因子
        self.scale = d_model ** -0.5
        
        # 存储最近的注意力权重
        self.last_attention = None
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        # 初始化查询投影
        nn.init.xavier_uniform_(self.query.weight)
        if self.query.bias is not None:
            nn.init.zeros_(self.query.bias)
        
        # 初始化专家键
        nn.init.xavier_uniform_(self.key)
        
        # 初始化门控网络
        for module in self.gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _compute_routing_loss(self, router_logits, expert_mask):
        """计算路由损失，包括负载均衡和辅助损失"""
        # 计算每个专家的使用量
        expert_usage = torch.sum(expert_mask, dim=(0, 1))  # [num_experts]
        total_tokens = expert_mask.size(0) * expert_mask.size(1)
        
        # 计算理想的均匀分布
        ideal_usage = torch.ones_like(expert_usage) * total_tokens / self.num_experts
        
        # 计算负载均衡损失 (使用KL散度)
        expert_usage_prob = expert_usage / total_tokens
        ideal_usage_prob = ideal_usage / total_tokens
        load_balancing_loss = F.kl_div(
            expert_usage_prob.log(), ideal_usage_prob,
            reduction='batchmean'
        )
        
        # 计算辅助损失 (鼓励稀疏的路由决策)
        router_probs = F.softmax(router_logits, dim=-1)
        aux_loss = torch.mean(
            router_probs * torch.log(router_probs + 1e-9)
        )
        
        # 合并损失
        total_loss = (
            self.router_z_loss_coef * load_balancing_loss +
            self.router_aux_loss_coef * aux_loss
        )
        
        return total_loss
        
    def forward(self, x):
        B, S, D = x.shape
        
        # 计算注意力分数
        q = self.query(x)  # [B, S, D]
        k = self.key  # [E, D]
        
        # 计算注意力权重
        attn_scores = torch.matmul(q, k.transpose(0, 1)) * self.scale  # [B, S, E]
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 存储注意力权重用于可视化
        self.last_attention = attn_weights.detach()
        
        # 获取top-k专家
        scores = attn_weights.mean(dim=1)  # [B, E]
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)  # [B, k]
        
        # 创建专家mask
        mask = torch.zeros_like(scores).scatter_(-1, top_k_indices, 1.0)  # [B, E]
        mask = mask.unsqueeze(1).expand(-1, S, -1)  # [B, S, E]
        
        # 计算路由损失
        router_loss = self._compute_routing_loss(attn_scores, mask)
        
        return mask, top_k_indices, router_loss
    
    def get_attention_weights(self):
        """返回最近的注意力权重"""
        return self.last_attention if self.last_attention is not None else None 