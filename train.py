import os
import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from src.modeling_esa import ESAConfig, ESAForCausalLM

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def prepare_dataset(config):
    """Load and preprocess dataset"""
    dataset = load_dataset(config["dataset"]["name"])
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["context"],
            truncation=True,
            padding="max_length",
            max_length=config["dataset"]["max_seq_len"]
        )
    
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["context", "question", "choices", "answer"]
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    return tokenized_datasets, data_collator

def initialize_model(config):
    """Initialize ESA model"""
    model_config = ESAConfig(
        hidden_size=config["model"]["hidden_size"],
        num_attention_heads=config["model"]["num_attention_heads"],
        num_hidden_layers=config["model"]["num_hidden_layers"],
        compress_dim=config["model"]["compress_dim"],
        initial_token_len=config["model"]["initial_token_len"],
        local_token_len=config["model"]["local_token_len"],
        top_k=config["model"]["top_k"],
        vocab_size=config["model"]["vocab_size"]
    )
    model = ESAForCausalLM(model_config)
    return model

def main():
    # Load configuration
    config = load_config("config.yaml")
    
    # Prepare dataset
    tokenized_datasets, data_collator = prepare_dataset(config)
    
    # Initialize model
    model = initialize_model(config)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config["training"]["save_dir"],
        overwrite_output_dir=True,
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        save_steps=1000,
        save_total_limit=2,
        logging_dir=config["training"]["log_dir"],
        logging_steps=100,
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=config["training"]["warmup_steps"],
        fp16=config["hardware"]["fp16"],
        gradient_accumulation_steps=config["hardware"]["gradient_accumulation_steps"],
        prediction_loss_only=True
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator
    )
    
    # Start training
    trainer.train()
    
    # Save final model
    trainer.save_model(config["training"]["save_dir"])

if __name__ == "__main__":
    main()