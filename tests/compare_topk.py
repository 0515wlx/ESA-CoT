import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modeling_esa import ESAConfig, ESAForCausalLM
from datasets import load_dataset
from transformers import AutoTokenizer

def load_longbench_data():
    """从Hugging Face加载LongBench-v2数据集"""
    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    return dataset

def generate_test_data(batch_size=2, seq_len=512, hidden_size=1536):
    """从LongBench生成测试数据"""
    dataset = load_longbench_data()
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
    
    # 获取前batch_size个样本
    samples = dataset[:batch_size]
    
    # 对context进行编码
    encoded = tokenizer(
        samples['context'],
        padding='max_length',
        max_length=seq_len,
        truncation=True,
        return_tensors='pt'
    )
    
    # 使用相同的输入作为queries和keys
    queries = encoded['input_ids'].unsqueeze(-1).expand(-1, -1, hidden_size)
    keys = queries.clone()
    
    return queries, keys

def compare_topk_methods():
    """对比两种top-k计算方法"""
    # 初始化模型
    config = ESAConfig(
        hidden_size=1536,
        top_k=64
    )
    model = ESAForCausalLM(config)
    
    # 生成测试数据
    queries, keys = generate_test_data()
    
    # 方法1: compute_importance_scores
    scores1 = model.compute_importance_scores(queries, keys)
    _, topk_indices1 = model.select_top_k_tokens(scores1, config.top_k)
    
    # 方法2: sliding_window_attention
    topk_indices2 = model.sliding_window_attention(queries, keys, config.top_k)
    
    # 计算指标
    batch_size = queries.size(0)
    overlap_ratios = []
    for i in range(batch_size):
        set1 = set(topk_indices1[i].tolist())
        set2 = set(topk_indices2[i].tolist())
        overlap = len(set1.intersection(set2)) / config.top_k
        overlap_ratios.append(overlap)
    
    # 输出结果
    print(f"Top-k tokens overlap ratio: {np.mean(overlap_ratios):.2%}")
    print(f"Top-k tokens overlap details: {overlap_ratios}")
    print(f"Method 1 indices: {topk_indices1.tolist()}")
    print(f"Method 2 indices: {topk_indices2.tolist()}")
    
    return {
        'overlap_ratio': np.mean(overlap_ratios),
        'overlap_details': overlap_ratios,
        'method1_indices': topk_indices1.tolist(),
        'method2_indices': topk_indices2.tolist()
    }

if __name__ == "__main__":
    print(compare_topk_methods())