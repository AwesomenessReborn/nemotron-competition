"""Validates a saved LoRA adapter against competition requirements."""
import json, argparse, sys
from pathlib import Path
from safetensors.torch import load_file


def validate(adapter_dir: str) -> bool:
    p = Path(adapter_dir)
    passed = True

    def check(condition, msg):
        nonlocal passed
        if not condition:
            print(f"  FAIL: {msg}")
            passed = False
        else:
            print(f"  OK:   {msg}")

    print(f"\nValidating adapter at: {p}\n")
    check((p / "adapter_config.json").exists(), "adapter_config.json exists")
    check((p / "adapter_model.safetensors").exists(), "adapter_model.safetensors exists")

    if (p / "adapter_config.json").exists():
        cfg = json.loads((p / "adapter_config.json").read_text())
        check(cfg.get("peft_type") == "LORA", f"peft_type == LORA (got {cfg.get('peft_type')})")
        check(cfg.get("r", 999) <= 32, f"rank <= 32 (got {cfg.get('r')})")
        check(
            cfg.get("task_type") in ["CAUSAL_LM", None, ""],
            f"task_type valid (got {cfg.get('task_type')})",
        )
        print(f"\n  base_model : {cfg.get('base_model_name_or_path', 'MISSING')}")
        print(f"  rank       : {cfg.get('r')}")
        print(f"  alpha      : {cfg.get('lora_alpha')}")
        print(f"  targets    : {cfg.get('target_modules')}")

    if (p / "adapter_model.safetensors").exists():
        try:
            weights = load_file(p / "adapter_model.safetensors")
            check(len(weights) > 0, f"safetensors loads ({len(weights)} tensors)")
        except Exception as e:
            check(False, f"safetensors loads: {e}")

    print(f"\n{'PASSED' if passed else 'FAILED'}\n")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    args = parser.parse_args()
    sys.exit(0 if validate(args.adapter_dir) else 1)
