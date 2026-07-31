"""Trainer pieces shared by ``train_sft_impl4.py`` and ``probe_loss_norm.py``.

``transformers`` is imported lazily inside the factories so the rest of the package
(and ``--help``) works in an environment without torch installed.
"""

from __future__ import annotations

import os
from pathlib import Path


def sequential_trainer_cls():
    """``Trainer`` whose train sampler is a ``SequentialSampler`` (PLAN §6).

    The HF default is ``RandomSampler``, which would shuffle the 24-pedagogy/8-general
    block layout away and turn the anchor back into an in-expectation constraint.
    The probe needs it too, so that the micro-batch partition it reasons about is the
    one the Trainer actually built.
    """
    from torch.utils.data import SequentialSampler
    from transformers import Trainer

    class SequentialTrainer(Trainer):
        def _get_train_sampler(self, *a, **kw):
            ds = a[0] if a else kw.get("train_dataset", None)
            return SequentialSampler(ds if ds is not None else self.train_dataset)

    return SequentialTrainer


class CheckpointGridRecorder:
    """Saves a PEFT adapter at each grid step and remembers which ones landed.

    Adapter only (PLAN §7): ~25 MB here (≈12M trainable params in bf16) against
    ~100 MB+ for a full trainer checkpoint that also writes fp32 Adam state.
    """

    def __init__(self, out_dir: str | os.PathLike, grid, log=print):
        self.out_dir = str(out_dir)
        self.grid = {int(s) for s in grid}
        self.saved: list[int] = []
        self.log = log

    def maybe_save(self, step: int, model) -> None:
        if step not in self.grid:
            return
        d = os.path.join(self.out_dir, f"ckpt-{step}")
        if os.path.isdir(d) and any(Path(d).glob("adapter_model*")):
            if step not in self.saved:
                self.saved.append(step)
            return
        model.save_pretrained(d)
        self.saved.append(step)
        self.log(f"  [ckpt-grid] saved adapter at step {step} -> {d}")


def checkpoint_grid_callback(out_dir, grid, log=print):
    """``(callback, recorder)`` — the recorder exposes ``saved`` after training."""
    from transformers import TrainerCallback

    rec = CheckpointGridRecorder(out_dir, grid, log=log)

    class _GridCallback(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kwargs):
            if model is not None:
                rec.maybe_save(int(state.global_step), model)
            return control

        def on_train_end(self, args, state, control, model=None, **kwargs):
            if model is not None:  # catches a final step that misses on_step_end
                rec.maybe_save(int(state.global_step), model)
            return control

    return _GridCallback(), rec


def loss_capture_callback(sink: list):
    """Appends every logged training loss to ``sink``. Used by the §5 probe."""
    from transformers import TrainerCallback

    class _LossCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                sink.append(float(logs["loss"]))
            return control

    return _LossCallback()
