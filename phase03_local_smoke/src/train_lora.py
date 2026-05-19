"""
Unsloth LoRA fine-tuning for Nemotron-H (Mamba-2 Transformer Hybrid).

Architecture notes:
- ~75% Mamba-2 SSM layers, ~25% standard attention (only 4 attention layers in 4B)
- Flash Attention 2 is NOT supported — Mamba layers have no Q/K/V
- LoRA must ONLY target attention/MLP projections, not Mamba SSM weights
- trust_remote_code=True is mandatory
- save_only_model=True avoids ~6GB optimizer state disk spikes per checkpoint
"""

import yaml, argparse, json, time
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from src.prompt_template import format_for_training


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_dataset(jsonl_path, tokenizer, max_seq_length):
    rows = load_jsonl(jsonl_path)
    formatted = [format_for_training(r) for r in rows]

    def apply_template(batch):
        texts = []
        for sys, usr, asst in zip(batch["system"], batch["user"], batch["assistant"]):
            messages = [
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
                {"role": "assistant", "content": asst},
            ]
            # enable_thinking=False: we supply the reasoning trace ourselves
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            texts.append(text)
        return {"text": texts}

    ds = Dataset.from_list(formatted)
    ds = ds.map(apply_template, batched=True, remove_columns=["system", "user", "assistant"])
    return ds


def main(cfg_path, train_path, val_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print(f"Loading model: {cfg['model_name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=cfg.get("load_in_4bit", False),
        trust_remote_code=True,
        # DO NOT set attn_implementation="flash_attention_2" —
        # Mamba-2 layers have no Q/K/V, FA2 dispatch doesn't exist for them
    )

    # Nemotron-H has only 4 standard attention layers.
    # q/k/v/o_proj target those; gate/up/down_proj target the MLP blocks.
    # Mamba SSM weights (in_proj, out_proj, x_proj, dt_proj, etc.) are excluded.
    target_modules = cfg.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print(f"Building datasets from {train_path} and {val_path}")
    train_ds = build_dataset(train_path, tokenizer, cfg["max_seq_length"])
    val_ds = build_dataset(val_path, tokenizer, cfg["max_seq_length"])
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
            num_train_epochs=cfg.get("num_train_epochs", 1),
            learning_rate=cfg.get("learning_rate", 2e-4),
            bf16=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=cfg.get("eval_steps", 50),
            save_strategy="steps",
            save_steps=cfg.get("save_steps", 100),
            save_total_limit=2,
            # save_only_model skips the ~6GB optimizer state per checkpoint.
            # Without it, two checkpoints × 6GB = 12GB disk spike causes corruption
            # (exact bug from professor's notebook at step 400).
            # Trade-off: cannot resume training from saved checkpoints.
            save_only_model=True,
            max_seq_length=cfg["max_seq_length"],
            dataset_text_field="text",
            report_to="none",
        ),
    )

    print("Starting training...")
    t0 = time.time()
    trainer.train()
    print(f"Training complete in {(time.time() - t0) / 60:.1f} min")

    final_path = output_dir / "final_adapter"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Adapter saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    args = parser.parse_args()
    main(args.config, args.train, args.val)
