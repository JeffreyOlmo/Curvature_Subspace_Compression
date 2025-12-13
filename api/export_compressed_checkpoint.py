import argparse
import json
from pathlib import Path

import torch

from curvature_subspace_compression.api.tiny_transformer_loader import TinyTransformerPair


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a keep-20% compressed TinyTransformer checkpoint.")
    parser.add_argument(
        "--comp-json",
        type=Path,
        required=True,
        help="Path to the v1 *_comp020.json artifact (contains the config and checkpoint pointer).",
    )
    parser.add_argument(
        "--baseline-ckpt",
        type=Path,
        required=False,
        default=None,
        help="Optional baseline .pt checkpoint (defaults to config.checkpoint_path).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Where to write the compressed model state_dict (.pt).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("curvature_subspace_compression/data"),
        help="Cache dir for datasets/tokenizer.",
    )
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    with args.comp_json.open() as f:
        data = json.load(f)
    cfg = data.get("config", {})
    ckpt = args.baseline_ckpt or Path(cfg.get("checkpoint_path"))

    pair = TinyTransformerPair(
        comp_json_path=args.comp_json,
        baseline_ckpt_path=ckpt,
        compressed_ckpt_path=args.out,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    pair.load()
    _ = pair.compressed()
    # Saved by TinyTransformerPair as part of compressed() call.
    torch.cuda.synchronize()
    print(f"Wrote compressed checkpoint to {args.out}")


if __name__ == "__main__":
    main()


