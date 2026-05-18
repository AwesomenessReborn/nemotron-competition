"""Validates and zips a LoRA adapter for Kaggle submission."""
import zipfile, argparse
from pathlib import Path
from src.validate_adapter import validate


def package(adapter_dir: str, out_path: str) -> None:
    adapter_dir = Path(adapter_dir)
    if not validate(str(adapter_dir)):
        raise RuntimeError("Adapter validation failed — fix before packaging.")

    required = ["adapter_config.json", "adapter_model.safetensors"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in required:
            src = adapter_dir / fname
            assert src.exists(), f"Missing required file: {fname}"
            zf.write(src, arcname=fname)

    print(f"Submission packaged -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--out", default="outputs/submissions/submission.zip")
    args = parser.parse_args()
    package(args.adapter_dir, args.out)
