"""
Build a zero-initialized LoRA adapter with correct shapes for the 30B Nemotron model.

The adapter is a valid no-op (lora_B=0, lora_A=0) so performance equals the base model.
This lets you make an initial Kaggle submission while the real training run is in flight.

Usage:
    conda run -n nemotron-train python phase04_cloud_train/make_placeholder_adapter.py
"""
import json
import torch
import safetensors.torch as st
import zipfile
from pathlib import Path
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

RANK = 32
LORA_ALPHA = 64
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
OUT_DIR = Path("phase04_cloud_train/outputs/adapters/cloud_30b/placeholder_adapter")
ZIP_OUT = Path("phase04_cloud_train/outputs/submissions/submission.zip")

def main():
    print(f"Loading 30B config (no weights)...")
    config = AutoConfig.from_pretrained(
        BASE_MODEL, trust_remote_code=True, cache_dir="/tmp/nemotron30b_config"
    )

    print("Instantiating model structure on meta device...")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    print("Collecting LoRA target shapes...")
    state_dict = {}
    n = 0
    for name, mod in model.named_modules():
        short = name.split(".")[-1]
        if short in TARGET_MODULES and hasattr(mod, "weight"):
            out_features, in_features = mod.weight.shape
            key_prefix = f"base_model.model.{name}"
            # lora_A: (rank, in_features) — kaiming-style but we use zeros for no-op
            state_dict[f"{key_prefix}.lora_A.weight"] = torch.zeros(RANK, in_features, dtype=torch.bfloat16)
            # lora_B: (out_features, rank) — zero init (standard LoRA init)
            state_dict[f"{key_prefix}.lora_B.weight"] = torch.zeros(out_features, RANK, dtype=torch.bfloat16)
            n += 1

    print(f"  {n} LoRA module pairs  ({len(state_dict)} total tensors)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving adapter weights -> {OUT_DIR}/adapter_model.safetensors")
    st.save_file(state_dict, OUT_DIR / "adapter_model.safetensors")

    adapter_config = {
        "architectures": None,
        "auto_mapping": None,
        "base_model_name_or_path": BASE_MODEL,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": RANK,
        "revision": None,
        "target_modules": sorted(TARGET_MODULES),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    (OUT_DIR / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))
    print(f"Saved adapter_config.json")

    size_mb = (OUT_DIR / "adapter_model.safetensors").stat().st_size / 1e6
    print(f"  adapter_model.safetensors: {size_mb:.1f} MB (zeros compress very well in zip)")

    ZIP_OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Packaging -> {ZIP_OUT}")
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(OUT_DIR / "adapter_config.json", arcname="adapter_config.json")
        zf.write(OUT_DIR / "adapter_model.safetensors", arcname="adapter_model.safetensors")

    zip_mb = ZIP_OUT.stat().st_size / 1e6
    print(f"  submission.zip: {zip_mb:.1f} MB")
    print()
    print(f"Done. Submit {ZIP_OUT} to Kaggle.")
    print("Performance will match the 30B base model (no-op adapter).")
    print("Replace with the real adapter after Phase 04 training completes.")


if __name__ == "__main__":
    main()
