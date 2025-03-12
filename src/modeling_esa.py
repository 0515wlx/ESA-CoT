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
    def __init__(self, base_model, config: ESAConfig):
        super().__init__(config)
        self.base_model = base_model
        
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
        top_k_values, top_k_indices = torch.topk(scores, k, dim=-1)
        return top_k_values, top_k_indices

    def compute_importance_scores(self, queries, keys, attention_mask=None):
        # Compute importance scores using compressed representations
        compressed_queries = self.query_compressor(queries)
        compressed_keys = self.key_compressor(keys)
        
        # Compute dot product attention scores
        scores = torch.matmul(
            compressed_queries,
            compressed_keys.transpose(-1, -2))
        
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
        
        return scores

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get hidden states
        hidden_states = outputs.hidden_states[-1]
        
        # Split tokens into initial, middle and local
        batch_size, seq_len, _ = hidden_states.size()
        initial_tokens = hidden_states[:, :self.config.initial_token_len, :]
        local_tokens = hidden_states[:, -self.config.local_token_len:, :]
        middle_tokens = hidden_states[
            :,
            self.config.initial_token_len:-self.config.local_token_len,
            :
        ]
        
        # Compute importance scores
        scores = self.compute_importance_scores(
            queries=local_tokens,
            keys=middle_tokens
        )
        
        # Select top-k middle tokens
        _, top_k_indices = self.select_top_k_tokens(
            scores,
            self.config.top_k
        )
        
        # Gather selected tokens
        selected_tokens = torch.gather(
            middle_tokens,
            1,
            top_k_indices.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        )
        
        # Combine tokens
        combined_tokens = torch.cat([
            initial_tokens,
            selected_tokens,
            local_tokens
        ], dim=1)
        
        # Compute final outputs
        logits = self.base_model.lm_head(combined_tokens)
        
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