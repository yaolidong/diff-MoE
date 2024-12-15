import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embd, top_k, num_experts):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embd, num_experts)
        self.noisy_topk = nn.Linear(n_embd, num_experts)

    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noisy_topk(mh_output)

        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        # softplus 是一个平滑函数,使得输出值在0到正无穷之间 softplus(x) = ln(1+exp(x))

        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices 