import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers import PretrainedConfig
from typing import Optional

class ESAConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size: int = 768,
        initial_token_len: int = 128,
        local_token_len: int = 256,
        top_k: int = 64,
        num_heads: int = 12,
        eps: float = 1e-5,
        initializer_range: float = 0.02,
        vocab_size: int = 50257,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.initial_token_len = initial_token_len
        self.local_token_len = local_token_len
        self.top_k = top_k
        self.num_heads = num_heads
        self.eps = eps
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size

class ESAForCausalLM(PreTrainedModel):
    def __init__(self, config: ESAConfig):
        super().__init__(config)
        
        # 基础模型组件
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size
        )
        
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0,
                std=self.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()

    def select_top_k_tokens(self, scores, k):
        # 根据重要性分数选择top-k tokens
        batch_size = scores.size(0)
        seq_len = scores.size(1)
        
        # Flatten scores to 2D [batch_size, seq_len]
        scores = scores.view(batch_size, -1)
        
        # Get top-k indices
        top_k_values, top_k_indices = torch.topk(scores, k, dim=-1)
        
        # Ensure indices are within valid range
        seq_len = scores.size(1)
        max_index = min(seq_len - 1, self.config.top_k - 1)
        top_k_indices = torch.clamp(top_k_indices, 0, max_index)
        
        # Reshape indices to match original dimensions
        top_k_indices = top_k_indices % (seq_len // k)
        
        return top_k_values, top_k_indices.view(batch_size, k)

    def compute_importance_scores(self, queries, keys, attention_mask=None):
        # 计算点积注意力分数
        scores = torch.matmul(
            queries,
            keys.transpose(-1, -2))
        
        # 通过隐藏维度的平方根缩放分数
        scores = scores / (self.config.hidden_size ** 0.5)
        
        # 如果提供了注意力掩码则应用
        if attention_mask is not None:
            # 为局部注意力创建因果掩码
            seq_len = queries.size(1)
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=queries.device))
            scores = scores.masked_fill(
                causal_mask == 0, float('-inf'))
            
            # 应用输入注意力掩码
            scores = scores.masked_fill(
                attention_mask.unsqueeze(1) == 0, float('-inf'))
        
        # 确保分数具有正确的维度
        batch_size = queries.size(0)
        local_len = queries.size(1)
        middle_len = keys.size(1)
        scores = scores.view(batch_size, local_len, middle_len)
        
        return scores

    def sliding_window_attention(self, queries, keys, window_size):
        """
        使用滑动窗口计算注意力分数
        Args:
            queries: 查询向量 [batch_size, seq_len, hidden_size]
            keys: 键向量 [batch_size, seq_len, hidden_size]
            window_size: 滑动窗口大小
        Returns:
            滑动窗口计算后的注意力分数 [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = queries.shape
        
        # 初始化select_tokens为前k个token
        select_tokens = queries[:, :window_size, :]
        
        # 初始化注意力分数矩阵
        scores = torch.zeros(
            batch_size, seq_len, seq_len,
            device=queries.device
        )
        
        # 从k~2k开始计算
        for i in range(window_size, seq_len, window_size):
            # 计算当前窗口范围
            window_start = max(0, i - window_size)
            window_end = min(seq_len, i + window_size)
            
            # 获取当前窗口的queries和keys
            window_queries = queries[:, window_start:window_end, :]
            window_keys = keys[:, window_start:window_end, :]
            
            # 计算select_tokens和窗口tokens的共同注意力分数
            combined_queries = torch.cat([select_tokens, window_queries], dim=1)
            combined_keys = torch.cat([select_tokens, window_keys], dim=1)
            
            # 计算联合注意力分数
            combined_scores = self.compute_importance_scores(
                queries=combined_queries,
                keys=combined_keys
            )
            
            # 从联合集合中选择top-k tokens
            _, top_k_indices = self.select_top_k_tokens(
                combined_scores, window_size)
            
            # 从联合集合中收集top-k tokens
            combined_tokens = torch.cat([select_tokens, window_queries], dim=1)
            select_tokens = torch.gather(
                combined_tokens,
                dim=1,
                index=top_k_indices.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
            )
            
            # 将当前窗口的分数更新到总分数矩阵中
            scores[:, window_start:window_end, window_start:window_end] = window_scores
            
        return scores

    def forward(self, input_ids, attention_mask=None, labels=None):
        # 生成随机隐藏状态作为占位符
        batch_size, seq_len = input_ids.shape
        hidden_states = torch.randn(
            batch_size,
            seq_len,
            self.config.hidden_size,
            device=input_ids.device
        )
        
        # 使用滑动窗口计算注意力分数
        scores = self.sliding_window_attention(
            queries=hidden_states,
            keys=hidden_states,
            window_size=self.config.top_k
        )
        
        # 选择top-k中间tokens
        _, top_k_indices = self.select_top_k_tokens(
            scores,
            self.config.top_k
        )
        
        # 从隐藏状态中收集中间tokens
        middle_tokens = torch.gather(
            hidden_states,
            dim=1,
            index=top_k_indices.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        )
        
        # 收集选定的tokens
        selected_tokens = torch.gather(
            middle_tokens,
            1,
            top_k_indices.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        )
        
        # 使用完整序列作为输出
        logits = self.lm_head(hidden_states)
        
        # 如果提供了标签则计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
        
        return {
            "logits": logits,
            "loss": loss
        }