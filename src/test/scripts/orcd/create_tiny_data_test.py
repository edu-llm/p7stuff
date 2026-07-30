import importlib.util
import json
from pathlib import Path

import numpy as np


def load_module():
    path = Path("src/scripts/orcd/create_tiny_data.py")
    spec = importlib.util.spec_from_file_location("tiny_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_create_data_is_deterministic(tmp_path):
    module = load_module()
    first = module.create_data(tmp_path / "a", train_tokens=4096, eval_tokens=1024, seed=7)
    second = module.create_data(tmp_path / "b", train_tokens=4096, eval_tokens=1024, seed=7)
    assert np.array_equal(
        np.memmap(first["train"], dtype=np.uint16),
        np.memmap(second["train"], dtype=np.uint16),
    )
    assert Path(first["eval"]).stat().st_size == 1024 * np.dtype(np.uint16).itemsize


def test_create_data_uses_gpt2_tokens_and_records_manifest(tmp_path):
    module = load_module()
    files = module.create_data(tmp_path, train_tokens=4096, eval_tokens=1024, seed=7)

    train = np.memmap(files["train"], dtype=np.uint16)
    eval_tokens = np.memmap(files["eval"], dtype=np.uint16)
    assert int(train.max()) < 50_257
    assert int(eval_tokens.max()) < 50_257
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == {
        "format": "raw-uint16-token-stream",
        "tokenizer": "gpt2",
        "seed": 7,
        "train_tokens": 4096,
        "eval_tokens": 1024,
    }
