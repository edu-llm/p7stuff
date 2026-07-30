import argparse
import json
from pathlib import Path

import numpy as np


def write_tokens(path: Path, count: int, seed: int) -> None:
    """
    Write a deterministic raw uint16 GPT-2 token stream.

    :param path: The output token-file path.
    :param count: The number of tokens to write.
    :param seed: The NumPy random seed.
    """
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.memmap(path, mode="w+", dtype=np.uint16, shape=(count,))
    array[:] = rng.integers(0, 50_257, size=count, dtype=np.uint16)
    array.flush()


def create_data(root: Path, *, train_tokens: int, eval_tokens: int, seed: int) -> dict[str, str]:
    """
    Create deterministic raw train and evaluation token files.

    :param root: The output directory.
    :param train_tokens: The number of training tokens to create.
    :param eval_tokens: The number of evaluation tokens to create.
    :param seed: The training-stream random seed. Evaluation uses the next seed.

    :returns: Paths to the generated training and evaluation token files.
    """
    train = root / "c4-train.00000-00099.npy"
    eval_path = root / "c4-validation.00000-00008.npy"
    write_tokens(train, train_tokens, seed)
    write_tokens(eval_path, eval_tokens, seed + 1)
    manifest = {
        "format": "raw-uint16-token-stream",
        "tokenizer": "gpt2",
        "seed": seed,
        "train_tokens": train_tokens,
        "eval_tokens": eval_tokens,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"train": str(train), "eval": str(eval_path)}


def main() -> None:
    """Create deterministic tiny data from command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=1_000_000)
    parser.add_argument("--eval-tokens", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(
        create_data(
            args.output,
            train_tokens=args.train_tokens,
            eval_tokens=args.eval_tokens,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
