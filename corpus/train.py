#!/usr/bin/env python3
"""
Rust CLI Documentation Fine-Tuning Script
Per RCDC-SPEC-001 Section 9.2

Uses QLoRA (4-bit quantization + LoRA) for efficient fine-tuning.
"""

import os
import sys
from pathlib import Path

# Check dependencies
def check_deps():
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    try:
        import peft
    except ImportError:
        missing.append("peft")
    try:
        import datasets
    except ImportError:
        missing.append("datasets")
    try:
        import bitsandbytes
    except ImportError:
        missing.append("bitsandbytes")

    if missing:
        print("Missing dependencies. Install with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

check_deps()

import torch
from datasets import Dataset
import pyarrow.parquet as pq
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# Configuration per spec Section 9.2
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # Use 7B for production
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LEARNING_RATE = 2e-4
EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
WARMUP_RATIO = 0.03

def load_corpus(path: str) -> Dataset:
    """Load corpus from parquet file."""
    table = pq.read_table(path)
    df = table.to_pandas()

    # Create training prompts
    def format_example(row):
        return {
            "text": f"### Input:\n{row['input']}\n\n### Documentation:\n{row['output']}"
        }

    df["text"] = df.apply(format_example, axis=1)
    return Dataset.from_pandas(df[["text"]])

def main():
    print("=" * 60)
    print("Rust CLI Documentation Fine-Tuning")
    print("=" * 60)

    # Paths
    corpus_path = Path(__file__).parent / "data/corpus/train.parquet"
    output_dir = Path(__file__).parent / "output/rust-cli-docs"

    print(f"\nCorpus: {corpus_path}")
    print(f"Output: {output_dir}")
    print(f"Model: {MODEL_NAME}")
    print(f"LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}")

    # Check CUDA
    if not torch.cuda.is_available():
        print("\nError: CUDA not available. GPU required for training.")
        sys.exit(1)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load corpus
    print("\nLoading corpus...")
    dataset = load_corpus(str(corpus_path))
    print(f"Loaded {len(dataset)} examples")

    # Split dataset
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=["text"])

    # QLoRA config (4-bit quantization)
    print("\nLoading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        report_to="none",  # Disable wandb
        seed=42,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    trainer.train()

    # Save
    print("\nSaving model...")
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
