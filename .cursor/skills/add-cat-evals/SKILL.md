---
name: add-cat-evals
description: >-
  Add ATLAS CAT (Computerized Adaptive Testing) checkpoint diagnostics to an
  OLMo-core training run. Converts a native OLMo-core checkpoint to HF format,
  runs the adaptive atlas_arc eval via vLLM, and writes theta/SE results to S3.
  Use when a training team wants to auto-run CAT diagnostics on checkpoints, add
  atlas_arc / adaptive-testing evals to a training script, or asks how to score
  checkpoints with ATLAS CAT.
disable-model-invocation: true
---

# Add CAT evals to a training run

Runs an ATLAS adaptive test (`atlas_arc`) on a checkpoint and writes results to a
fixed S3 prefix. Designed to be called from an OLMo-core training script right
after a checkpoint is saved.

## What it does

1. Converts a native OLMo-core checkpoint (`model_and_optim/` + `config.json`) to
   HF format (`config.json` + `*.safetensors`). CPU-only for standard dense archs.
2. Boots a local vLLM server on the HF checkpoint.
3. Runs the adaptive `atlas_arc` CAT (picks 8–40 ARC-Challenge items until SE ≤ stop).
4. Writes `atlas_arc_results.json` (theta, se, pirt_accuracy, n_items, selected ids),
   `pipeline_provenance.json`, and `worker.log` to S3, plus a `_READY` marker.

Scope today: **ARC-Challenge only** (one ability signal, not the full 5-benchmark
ATLAS profile). It is an old-task-retention / ability probe. It does **not** score
pedagogy quality, instruction-following, or math answer formatting — if a run needs
those, CAT does not replace them.

## Checkpoint kinds

`run_cat_diagnostic.sh` accepts four shapes and picks the path by inspecting the dir:

| Detected | Marker | What happens |
|---|---|---|
| native OLMo-core | `model_and_optim/` | converted with `src/examples/huggingface/convert_checkpoint_to_hf.py` |
| HF format | `config.json` + weights | used as-is (no conversion) |
| PEFT / LoRA adapter | `adapter_config.json` | merged onto its base model (`merge_and_unload`), then used |
| HF hub id | not a local path | used as-is |

The adapter path exists because the p7 POC's SFT runs save LoRA adapters, not native
checkpoints. It reads `base_model_name_or_path` from `adapter_config.json`; pass
`--base-model` to override or to supply it when absent.

## Requirements

**Conversion node** (the training node, CPU):
- `ai2-olmo-core==2.4.0` (already present in an OLMo-core env) + `transformers`
- `peft` as well, for the adapter path
- RAM ≈ model size at bf16 (~2 GB per 1B params), ~2× if validation runs
- Disk ≈ 2× model size (input shards + HF output)
- Standard **dense** OLMo-2/OLMo-3 arch. MoE / fused / flash-attention checkpoints
  force conversion onto GPU and are not covered by the cheap path.

**Diagnostic (this repo's env, 1 GPU)**:
- `olmo-eval` installed with vLLM extras (`uv sync --extra vllm --extra hf`)
- 1× GPU with ~24 GB VRAM (fits ≤ ~8B at bf16). Larger models need a bigger GPU
  or tensor-parallel (`--tp N`).
- AWS creds with `s3:PutObject` on the target results prefix.

Nothing is uploaded to the HuggingFace Hub. "HF" means the on-disk file format;
weights and results stay local / in your S3.

## Quick start (in-process, from a training script)

After the checkpoint lands at `${CKPT_DIR}` (a dir containing `config.json` and
`model_and_optim/`):

```bash
bash .cursor/skills/add-cat-evals/scripts/run_cat_diagnostic.sh \
  --checkpoint "${CKPT_DIR}" \
  --run-id "${EXP}-step${STEP}" \
  --s3-out "s3://YOUR_BUCKET/atlas_cat"
```

Run it **backgrounded** so it doesn't block training:

```bash
nohup bash .cursor/skills/add-cat-evals/scripts/run_cat_diagnostic.sh \
  --checkpoint "${CKPT_DIR}" --run-id "${EXP}-step${STEP}" \
  --s3-out "s3://YOUR_BUCKET/atlas_cat" >/dev/null 2>&1 &
```

For a LoRA adapter (the p7 POC case):

```bash
bash .cursor/skills/add-cat-evals/scripts/run_cat_diagnostic.sh \
  --checkpoint runs/A3/ckpt-160 --base-model allenai/OLMo-2-0425-1B-Instruct \
  --run-id "A3-step160" --s3-out "s3://YOUR_BUCKET/atlas_cat"
```

`--run-id` must match `[A-Za-z0-9._-]+`. Results land at `<s3-out>/<run-id>/`.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | required | OLMo-core checkpoint dir (or an HF dir / adapter dir / HF id) |
| `--run-id` | required | results subdirectory; `[A-Za-z0-9._-]+` |
| `--s3-out` | required | `s3://bucket/prefix` root for results |
| `--tokenizer` | (from config) | HF tokenizer id if not resolvable from checkpoint config |
| `--base-model` | (from adapter config) | base model a PEFT adapter is merged onto |
| `--se-stop` | `0.3` | CAT early-stop standard error |
| `--min-items` | `8` | CAT floor |
| `--max-items` | `40` | CAT cap |
| `--tp` | `1` | vLLM tensor-parallel size (raise for large models) |
| `--skip-convert` | off | checkpoint is already HF format / an HF id |
| `--keep-hf` | off | keep the converted `-hf` dir (default: temp, removed after) |
| `--skip-preflight` | off | skip the "is olmo-eval runnable" check that guards the conversion |
| `--validate-conversion` | off | run olmo-core's numerical validation (~2× RAM, much slower) |
| `--converter` | (auto) | override `convert_checkpoint_to_hf.py` location |
| `--olmo-eval-cmd` | (auto) | how to invoke olmo-eval, e.g. `"uv run olmo-eval"` |
| `--work-dir` | (mktemp) | staging dir for HF weights + results |
| `--dry-run` | off | print the plan without running |

## Output location

```
<s3-out>/<run-id>/
  atlas_arc_results.json    # theta, se, pirt_accuracy, n_items, selected_question_ids
  pipeline_provenance.json  # checkpoint, run_id, git_sha, args, versions, timings
  worker.log
  _READY                    # written last; poll for this to know it's done
```

On failure `worker.log` is still uploaded but `_READY` is **not**, so a poller never
mistakes a crashed run for a finished one.

## Verify locally first (no GPU, no AWS)

```bash
bash .cursor/skills/add-cat-evals/scripts/run_cat_diagnostic.sh \
  --checkpoint runs/A3/ckpt-160 --run-id smoke --s3-out s3://bucket/atlas_cat --dry-run
```

`--dry-run` resolves the checkpoint kind, the converter, the olmo-eval command and the
destination, then exits before touching a GPU or S3. The end-to-end pytest for the CAT
pipeline (`tests/adaptive/test_atlas_cat_checkpoint_pipeline.py`) lives in the
olmo-eval / AdaptiveTesting repo, not here.

## How it maps to olmo-eval

The diagnostic step is exactly:

```bash
uv run olmo-eval run-external \
  -m "${HF_CKPT}" -e atlas_arc --provider vllm_server \
  -a "se_stop=0.3" -a "min_items=8" -a "max_items=40" \
  -O "${OUT}"
```

`atlas_arc` only runs through `run-external` (vLLM), which needs HF-format weights —
hence the conversion step. Native OLMo-core weights are not loadable by vLLM directly.
`run-external --provider vllm_server` owns the server lifecycle; the script does not
boot vLLM itself.

olmo-eval's output layout is version-dependent, so the script locates the payload by
content (the JSON object carrying a `theta`) and normalises it to
`atlas_arc_results.json`. If it finds none it fails and prints the directory listing
rather than uploading an empty result.

## Additional resources

- Conversion details, memory/timing, and architecture caveats: see [CONVERSION.md](CONVERSION.md)
- Scoring a whole checkpoint grid (the p7 POC's `runs/<arm>/ckpt-*`):
  `p7/POC/atlas_cat/score_checkpoints.sh`
- Separate-GPU-worker path (launch an EC2 box per checkpoint instead of in-process):
  `AdaptiveTesting/scripts/atlas_cat_diagnose/` (`launch_g6.sh`, `run_worker.sh`)
