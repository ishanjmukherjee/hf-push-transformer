import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="tiny-shakespeare-gpt1l")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # 1. Load Tiny Shakespeare text (~1 MB) directly from GitHub.
    dataset = load_dataset(
        "text",
        data_files={
            "train": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        },
        split="train",
    )

    # 2. Tokenizer: byte-level BPE (GPT-2) works well for English text.
    tokenizer_name = "gpt2"  # reuse GPT-2 vocab instead of training one from scratch
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.model_max_length = args.block_size
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

    tokenized = dataset.map(
        lambda examples: tokenizer(examples["text"], return_special_tokens_mask=False),
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=["text"],
    )

    # 3. Group tokens into fixed-length blocks for efficient causal-LM training.
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // args.block_size) * args.block_size
        result = {
            k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()  # teacher-forcing
        return result

    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
    )

    # 4. Configure a *tiny* GPT-like model with **one** transformer layer.
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=args.block_size,
        n_ctx=args.block_size,
        n_embd=256,
        n_layer=1,  # <-single layer!
        n_head=8,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)

    # 5. Trainer setup.
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        push_to_hub=False,  # Save locally for now; push after inspecting.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=data_collator,
    )

    # 6. Train!
    trainer.train()

    # 7. Save artifacts ready for the Hub.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nTraining complete. Model & tokenizer saved to '{args.output_dir}'.")


if __name__ == "__main__":
    main()
