import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers import PretrainedConfig
from typing import Optional

class ESAConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size: int = 768,
        compress_dim: int = 128,
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
        self.compress_dim = compress_dim
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
        
        # Base model components
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size
        )
        
        # Compression layers
        self.query_compressor = nn.Linear(
            config.hidden_size,
            config.compress_dim
        )
        self.key_compressor = nn.Linear(
            config.hidden_size,
            config.compress_dim
        )
        
        # Initialize weights
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
        # Select top-k tokens based on importance scores
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
        # Compute importance scores using compressed representations
        compressed_queries = self.query_compressor(queries)
        compressed_keys = self.key_compressor(keys)
        
        # Compute dot product attention scores
        scores = torch.matmul(
            compressed_queries,
            compressed_keys.transpose(-1, -2))
        
        # Scale scores by sqrt of compressed dimension
        scores = scores / (self.config.compress_dim ** 0.5)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Create causal mask for local attention
            seq_len = queries.size(1)
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=queries.device))
            scores = scores.masked_fill(
                causal_mask == 0, float('-inf'))
            
            # Apply input attention mask
            scores = scores.masked_fill(
                attention_mask.unsqueeze(1) == 0, float('-inf'))
        
        # Ensure scores have correct dimensions
        batch_size = queries.size(0)
        local_len = queries.size(1)
        middle_len = keys.size(1)
        scores = scores.view(batch_size, local_len, middle_len)
        
        return scores

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Generate random hidden states as placeholder
        batch_size, seq_len = input_ids.shape
        hidden_states = torch.randn(
            batch_size,
            seq_len,
            self.config.hidden_size,
            device=input_ids.device
        )
        
        # Use full sequence for both queries and keys
        scores = self.compute_importance_scores(
            queries=hidden_states,
            keys=hidden_states
        )
        
        # Select top-k middle tokens
        _, top_k_indices = self.select_top_k_tokens(
            scores,
            self.config.top_k
        )
        
        # Gather middle tokens from hidden states
        middle_tokens = torch.gather(
            hidden_states,
            dim=1,
            index=top_k_indices.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        )
        
        # Gather selected tokens
        selected_tokens = torch.gather(
            middle_tokens,
            1,
            top_k_indices.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        )
        
        # Use full sequence for output
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
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