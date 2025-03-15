import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers import PretrainedConfig
from typing import Optional

class ESAConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size: int = 1536,
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

    def select_attention(self, queries, keys):
        """
        选择token并计算注意力分数
        Args:
            queries: 查询向量 [batch_size, seq_len, hidden_size]
            keys: 键向量 [batch_size, seq_len, hidden_size]
        Returns:
            选择后的token [batch_size, top_k, hidden_size]
        """
        # 计算重要性分数
        scores = self.compute_importance_scores(queries, keys)
        
        # 选择top-k tokens
        _, top_k_indices = self.select_top_k_tokens(scores, self.config.top_k)
        
        # 收集选择的tokens
        select_tokens = torch.gather(
            queries,
            dim=1,
            index=top_k_indices.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        )
        
        return select_tokens

    def __init__(self, config: ESAConfig):
        super().__init__(config)
        
        # 基础模型组件
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.initial_token_len, config.hidden_size)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=4*config.hidden_size,
            dropout=0.1,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size
        )
        
        # 初始化权重
        self.apply(self._init_weights)

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
        
        return top_k_indices

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        
        # 生成输入嵌入
        input_embeddings = self.embedding(input_ids)
        
        # 添加位置编码
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        
        hidden_states = input_embeddings + position_embeddings
        
        # 通过Transformer编码器
        hidden_states = self.encoder(hidden_states.transpose(0, 1)).transpose(0, 1)
        
        # 选择token并计算注意力
        hidden_states = self.select_attention(
            queries=hidden_states,
            keys=hidden_states
        )
        
        # 计算logits
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