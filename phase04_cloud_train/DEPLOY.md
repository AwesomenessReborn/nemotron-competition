# Phase 4 — Cloud Deployment Guide

## GPU Requirements

| GPU | VRAM | Verdict |
|-----|------|---------|
| H100 80 GB | 80 GB | Recommended — comfortable headroom |
| A100 80 GB | 80 GB | Works — ~75 GB peak with grad-ckpt |
| 2× A6000 48 GB | 96 GB | Works — needs multi-GPU config |
| A100 40 GB | 40 GB | No — model weights alone are 60 GB |

> The 30B BF16 model weights = 60 GB. With gradient checkpointing and LoRA-only optimizer states, peak VRAM is ~70–75 GB.
> If you're constrained to a 40 GB GPU, set `load_in_4bit: true` in `cloud_30b.yaml` (reduces to ~15 GB but slight quality loss).

**Recommended providers**: Lambda Labs, RunPod, Vast.ai, CoreWeave  
**Estimated training time**: ~3–5 hours on H100, ~5–7 hours on A100  
**Estimated cost**: $6–$20 depending on provider and GPU

---

## Step-by-Step Deployment

### 1. Provision the VM

Rent a GPU instance with:
- Ubuntu 22.04
- CUDA 12.x pre-installed (most cloud images include this)
- At least 100 GB disk (model download ~60 GB + checkpoints)
- SSH access

### 2. Clone the repo

```bash
git clone https://github.com/AwesomenessReborn/nemotron-competition.git
cd nemotron-competition
```

### 3. Run setup (installs conda, PyTorch, unsloth, mamba-ssm)

```bash
bash phase04_cloud_train/setup.sh
```

This takes 15–25 min. mamba-ssm compiles from source — nvcc must be in PATH.
Most cloud CUDA images have it at `/usr/local/cuda/bin/nvcc`; if not:

```bash
export PATH=/usr/local/cuda/bin:$PATH
bash phase04_cloud_train/setup.sh
```

### 4. HuggingFace login + model license

```bash
conda run -n nemotron-train huggingface-cli login
```

Then accept the model license at:
https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

(Click "Agree and access repository" while logged in with the same HF account)

### 5. Upload training data

The JSONL files are gitignored. Upload from your local machine:

```bash
# From your LOCAL machine (adjust IP/user):
scp phase02_data_generation/data/merged/train.jsonl  user@VM_IP:~/nemotron-competition/phase02_data_generation/data/merged/
scp phase02_data_generation/data/merged/val.jsonl    user@VM_IP:~/nemotron-competition/phase02_data_generation/data/merged/
```

Or create the merged directory on the VM first:
```bash
# On the VM:
mkdir -p phase02_data_generation/data/merged
```

**Alternative**: upload to an S3/GCS bucket and `aws s3 cp` / `gsutil cp` on the VM.

### 6. Run training

```bash
# From repo root on the VM:
bash phase04_cloud_train/run_training.sh
```

This runs preflight checks (env, data, HF auth, VRAM), then starts training.
Monitor live:
```bash
tail -f phase04_cloud_train/outputs/logs/train.log
```

Checkpoints save every 200 steps to `phase04_cloud_train/outputs/adapters/cloud_30b/`.

### 7. Download the adapter

After training completes, download the final adapter to your local machine:

```bash
# From your LOCAL machine:
scp -r user@VM_IP:~/nemotron-competition/phase04_cloud_train/outputs/adapters/cloud_30b/final_adapter \
    phase04_cloud_train/outputs/adapters/cloud_30b/
```

---

## Config Tuning

Key knobs in `phase04_cloud_train/configs/cloud_30b.yaml`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lora_rank` | 32 | Higher rank = more capacity. 64 if VRAM allows |
| `lora_alpha` | 64 | Set to 2× rank for standard scaling |
| `learning_rate` | 0.0001 | Lower than 4B run (larger model = smaller LR) |
| `gradient_accumulation_steps` | 8 | Effective batch = 8 samples |
| `load_in_4bit` | false | Set true if <80 GB VRAM |
| `save_only_model` | true | Skips optimizer state (~30 GB/checkpoint saved) |

---

## Resume from Checkpoint

If training is interrupted, just re-run `run_training.sh`. The training script has resume logic: it loads already-completed IDs and skips them.

---

## Troubleshooting

**`mamba_ssm` import fails after setup**
```bash
export PATH=/usr/local/cuda/bin:$PATH
conda run -n nemotron-train pip install \
    "mamba-ssm @ git+https://github.com/state-spaces/mamba.git" \
    --no-build-isolation --force-reinstall
```

**OOM during training**
- Set `load_in_4bit: true` in `cloud_30b.yaml`
- Or reduce `max_seq_length` to 1024
- Or use 2-GPU setup with `accelerate launch`

**Model download fails / 401 error**
- Ensure HF token has read access
- Accept model license on the HuggingFace website while logged in
- Re-run `huggingface-cli login`

**`trust_remote_code` warning**
- Expected — Nemotron-H uses custom modeling code. Safe to proceed.
