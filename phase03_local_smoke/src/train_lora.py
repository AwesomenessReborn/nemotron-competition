"""
Unsloth LoRA fine-tuning for Nemotron-H (Mamba-2 Transformer Hybrid).

ARCHITECTURE NOTES — read before modifying:
- ~75% Mamba-2 SSM layers, ~25% standard attention (4 attention layers in 4B)
- Flash Attention 2 NOT supported — Mamba layers have no Q/K/V
- DO NOT set attn_implementation='flash_attention_2'
- LoRA targets: attention + MLP projections only, NOT Mamba layer weights
- trust_remote_code=True is MANDATORY
- save_only_model=True prevents optimizer state disk explosion
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

sys.path.insert(0, str(Path(__file__).parent))
from prompt_template import format_for_training


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dataset(jsonl_path, tokenizer):
    rows = load_jsonl(jsonl_path)
    texts = []
    for row in rows:
        parts = format_for_training(row)
        messages = [
            {"role": "system",    "content": parts["system"]},
            {"role": "user",      "content": parts["user"]},
            {"role": "assistant", "content": parts["assistant"]},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        texts.append({"text": text})
    return Dataset.from_list(texts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val",   required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print(f"Model:      {cfg['model_name']}")
    print(f"Train data: {args.train}")
    print(f"Val data:   {args.val}")
    print(f"Output:     {cfg['output_dir']}")
    print(f"{'='*60}\n")

    # Load model
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
        dtype=None,
        trust_remote_code=True,
        # DO NOT pass attn_implementation — Mamba-2 layers don't support FA2
    )
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # Attach LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        use_gradient_checkpointing="unsloth",
        bias="none",
        random_state=42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)\n")

    # Build datasets
    train_ds = build_dataset(args.train, tokenizer)
    val_ds   = build_dataset(args.val,   tokenizer)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}\n")

    output_dir = cfg["output_dir"]
    final_adapter_dir = str(Path(output_dir) / "final_adapter")

    sft_cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        bf16=True,
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_only_model=cfg["save_only_model"],
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        report_to="none",
        logging_steps=1,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_cfg,
    )

    print("Starting training...\n")
    t_train = time.time()
    trainer.train()
    elapsed = time.time() - t_train
    print(f"\nTraining complete in {elapsed/60:.1f} min")

    # Save final adapter
    Path(final_adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    # Verify outputs
    print(f"\nChecking adapter files in {final_adapter_dir}:")
    expected = ["adapter_config.json", "adapter_model.safetensors"]
    all_ok = True
    for fname in expected:
        p = Path(final_adapter_dir) / fname
        if p.exists():
            print(f"  OK {fname}  ({p.stat().st_size / 1024:.0f} KB)")
        else:
            print(f"  MISSING {fname}")
            all_ok = False

    if all_ok:
        print("\nSMOKE TRAINING PASSED — adapter saved successfully.")
    else:
        print("\nWARNING: some adapter files are missing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
