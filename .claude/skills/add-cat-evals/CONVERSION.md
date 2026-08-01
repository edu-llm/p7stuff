# Checkpoint conversion for CAT

`atlas_arc` runs through `olmo-eval run-external`, which serves the model with vLLM.
vLLM cannot load native OLMo-core weights, so every non-HF checkpoint has to be
converted first. This is the step that decides whether the diagnostic is cheap
(minutes on the training node's CPU) or expensive (a GPU and a wait).

## The three input shapes

### 1. Native OLMo-core (`model_and_optim/` + `config.json`)

Converted by this repo's own CLI wrapper:

```bash
python src/examples/huggingface/convert_checkpoint_to_hf.py \
  -i "${CKPT_DIR}" -o "${CKPT_DIR}-hf" --dtype bfloat16 --skip-validation
```

The real logic is `olmo_core.nn.hf.convert_checkpoint.convert_checkpoint_to_hf`; the
script is a thin wrapper that reads `model` and `dataset.tokenizer` out of the
experiment config in the checkpoint dir. If that config is missing, conversion cannot
proceed — there is nothing to reconstruct the architecture from.

`run_cat_diagnostic.sh` passes `--skip-validation` by default. Validation loads a
second copy of the model and compares logits, which roughly doubles peak RAM and adds
most of the runtime. Turn it on with `--validate-conversion` when you are converting a
new architecture for the first time; leave it off for the hundredth checkpoint of a run
you have already validated once.

`--dtype bfloat16` is the default for a reason (allenai/olmo-cookbook#60): fp32 output
doubles disk and buys nothing, since vLLM will serve bf16 anyway.

**Tokenizer.** Normally resolved from the experiment config. When it isn't, pass
`--tokenizer <hf-id>`; without it the HF dir has no tokenizer and vLLM will fail to
serve it.

### 2. Already HF format

Nothing to do. `--skip-convert` is accepted but unnecessary — the script detects
`config.json` without `model_and_optim/` and skips conversion on its own.

### 3. PEFT / LoRA adapter (`adapter_config.json`)

Not in the upstream skill; added because the p7 POC's SFT runs save ~25 MB adapters
rather than full checkpoints, and those are what the retired POC evals graded.

The script loads the base model at bf16, applies the adapter, calls
`merge_and_unload()`, and writes a full HF model. That means:

* **Peak RAM ≈ base model at bf16** (~2 GB per 1B params), plus the adapter. The merge
  is in-place on the base weights, so there is no second full copy.
* **Disk ≈ one full model per checkpoint scored.** An 11-point checkpoint grid is
  11 × ~2.4 GB of transient HF weights for a 1B model. The script removes each one
  after the CAT finishes unless `--keep-hf` is passed. Score sequentially, not in
  parallel, unless you have the disk.
* The base model id comes from `adapter_config.json:base_model_name_or_path`. Pass
  `--base-model` to override it — and do override it if the adapter was trained against
  a local path that no longer exists.
* **The merged model's chat template comes from the base tokenizer.** For the POC that
  is correct: `allenai/OLMo-2-0425-1B-Instruct` ships the template that training used.
  A base checkpoint without a template will make vLLM format prompts differently from
  training, and the script warns when it sees that.

## Memory and timing, roughly

Per checkpoint, dense arch, conversion on CPU:

| Model | RAM (no validation) | RAM (validation) | Disk (transient) | Convert time |
|---|---|---|---|---|
| 190M | ~1 GB | ~2 GB | ~0.8 GB | seconds |
| 1B | ~3 GB | ~6 GB | ~5 GB | ~1 min |
| 7B | ~16 GB | ~32 GB | ~30 GB | ~5-10 min |
| 32B | ~70 GB | not on CPU | ~130 GB | ~30 min+ |

The CAT itself is 8-40 ARC-Challenge items against a vLLM server. Its cost has **not been
measured here** — olmo-eval is not installed in this repo, so the table above covers
conversion only. Expect server startup to dominate a short run, and expect adaptive
stopping to end an unambiguous checkpoint near `--min-items`; both are inferences from the
item counts in SKILL.md, not timings. Take the real figures from the first run
(`pipeline_provenance.json:timings_seconds.eval`) and replace this paragraph.

## Architectures that break the cheap path

The CPU-only route assumes a **standard dense** OLMo-2 / OLMo-3 transformer. These do
not fit it:

* **MoE** — expert weights need the GPU converter and `--moe-capacity-factor` tuning to
  avoid false validation failures.
* **Hybrid (GDN + attention)** — supported by the converter, but saved as raw
  `config.json` + `model.safetensors` rather than through `save_pretrained()`, so
  downstream tooling that expects a normal HF dir may need adjusting.
* **Fused / flash-attention-specific weight layouts** — need conversion on the device
  the kernels were built for.

For these, run conversion with `--device cuda` (pass through with `--converter` plus a
wrapper, or convert out of band and hand the script the HF dir with `--skip-convert`).

## Failure modes worth recognising

| Symptom | Cause |
|---|---|
| `Experiment config not found, cannot convert` | `-i` is not an OLMo-core checkpoint dir, or `config.json` was not copied alongside `model_and_optim/` |
| vLLM: no chat template | HF dir has no tokenizer, or a base (non-Instruct) tokenizer was used |
| OOM during conversion | validation is on (`--validate-conversion`), or the arch needs the GPU path |
| `no atlas_arc result with a 'theta' field` | olmo-eval ran but wrote a layout the normaliser did not recognise — read the printed listing and the `-O` dir |
| `_READY` never appears | the run failed; `worker.log` is uploaded anyway, read it in S3 |
