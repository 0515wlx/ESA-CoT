import unittest
import torch
from src.modeling_esa import ESAConfig, ESAForCausalLM
from unittest.mock import MagicMock

class TestESAModel(unittest.TestCase):
    def setUp(self):
        """Initialize test configuration and model"""
        self.config = ESAConfig(
            hidden_size=768,
            num_heads=12,
            initial_token_len=128,
            local_token_len=256,
            top_k=64,
            eps=1e-5,
            initializer_range=0.02,
            vocab_size=50257
        )
        
        self.model = ESAForCausalLM(self.config)

    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.config.hidden_size, 768)
        self.assertEqual(self.model.config.num_heads, 12)

    def test_compute_importance_scores(self):
        """Test importance score computation"""
        batch_size = 2
        seq_len = 512
        hidden_size = 768
        
        queries = torch.randn(batch_size, seq_len, hidden_size)
        keys = torch.randn(batch_size, seq_len, hidden_size)
        
        scores = self.model.compute_importance_scores(queries, keys)
        self.assertEqual(scores.shape, (batch_size, seq_len, seq_len))

    def test_attention_mechanism(self):
        """Test attention mechanism"""
        batch_size = 2
        seq_len = 512
        hidden_size = 768
        
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        outputs = self.model(input_ids, attention_mask)
        self.assertIn("logits", outputs)
        self.assertEqual(outputs["logits"].shape, 
                        (batch_size, seq_len, 50257))

    def test_forward_pass(self):
        """Test forward pass"""
        batch_size = 2
        seq_len = 512
        
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        outputs = self.model(input_ids, attention_mask)
        self.assertIn("logits", outputs)
        self.assertIsInstance(outputs["logits"], torch.Tensor)

    def test_training_step(self):
        """Test training step with labels"""
        batch_size = 2
        seq_len = 512
        
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 50257, (batch_size, seq_len))
        
        outputs = self.model(input_ids, attention_mask, labels)
        self.assertIn("loss", outputs)
        self.assertIsInstance(outputs["loss"], torch.Tensor)

if __name__ == "__main__":
    unittest.main()