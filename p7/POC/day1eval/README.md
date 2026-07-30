# day1eval — MRBench tutor-generation eval

Run small open LLMs **as math tutors** over the
[MRBench](https://github.com/kaushal0494/UnifyingAITutorEvaluation) benchmark,
then score the generated responses with an LLM judge. Runs on any GPU via
**vLLM** — the easiest path is the **Google Colab notebook** (`colab_eval.ipynb`).

> **Just want to run it?** Open `colab_eval.ipynb` in Colab, set a GPU runtime,
> add your `PROMPTLENS_API_KEY`, and Run All. It does all 4 configs end-to-end.
> The `.py` files below are the equivalent CLI for a standalone GPU box.

- **Default model:** `allenai/OLMo-2-0425-1B-Instruct` (post-trained OLMo-2 1B)
- **Alternative:** `Qwen/Qwen3-1.7B` (hybrid reasoning model)
- **Also available:** `allenai/OLMo-2-0425-1B` base (`--model olmo-base`)
- **Task:** given a MRBench dialogue whose last turn is a student mistake,
  generate the next **tutor** turn. Output responses can be scored/annotated
  later against the 8 MRBench pedagogical dimensions.

## Files

| File | Purpose |
|------|---------|
| **`colab_eval.ipynb`** | Single-turn MRBench eval: 4 configs, next-turn generation + 8-dimension judging + table. |
| `config.py` | Model registry, dataset URLs, generation/engine + **judge** defaults. |
| `data.py` | Download MRBench splits; parse `conversation_history` into turns. |
| `prompts.py` | Build the tutor-generation prompt (system + conversation). |
| `generate.py` | vLLM runner — load model, batch-generate, save JSON. |
| `scoring.py` | 8-dimension MRBench rubric + judge prompt/parse/aggregate. |
| `llm_client.py` | OpenAI-compatible PromptLens gateway client (retries 429/5xx). |
| `score.py` | LLM-as-a-judge scorer — rate generated responses, aggregate. |
| `run.sh` | Thin launcher for the CLI. |
| `requirements.txt` | vLLM + transformers (generation); requests + dotenv (scoring). |

## Colab (recommended)

`colab_eval.ipynb` is self-contained — no repo files needed on Colab, just the
notebook. It runs **4 configurations**:

| # | model | system prompt |
|---|-------|---------------|
| 1 | OLMo-2-1B-Instruct | baseline |
| 2 | Qwen3-1.7B | baseline |
| 3 | OLMo-2-1B-Instruct | custom pedagogical (hint-ladder / Socratic) |
| 4 | Qwen3-1.7B | custom pedagogical |

Steps: upload to Colab → `Runtime → Change runtime type → GPU` → add
`PROMPTLENS_API_KEY` as a Colab **Secret** (or paste when prompted) → **Run all**.
It generates with vLLM, scores every response with `gpt-5.6-sol`, and prints a
per-dimension comparison table. Set `NUM_DIALOGUES = 10` in the config cell for a
quick smoke test before the full 186-dialogue run.

## CLI (standalone GPU box)

```bash
pip install -r requirements.txt

# Default: OLMo-2-1B-Instruct over MRBench V1 (all dialogues)
python generate.py --model olmo --dataset V1

# Qwen3-1.7B instead
python generate.py --model qwen --dataset V1

# See the registry
python generate.py --list-models
```

Or via the launcher — `./run.sh <model> <dataset> <limit> [extra args…]`:

```bash
./run.sh olmo V1 20                 # 20-dialogue smoke test
./run.sh qwen V1 0 --thinking       # full run, Qwen3 reasoning traces on
```

Output lands in `outputs/<model>_<dataset>.json` as `{ meta, records[] }`,
each record containing the source dialogue plus `generated_response`.

## Models

| key | HF id | notes |
|-----|-------|-------|
| `olmo` *(default)* | `allenai/OLMo-2-0425-1B-Instruct` | Post-trained (SFT+DPO+RLVR); has a chat template. |
| `olmo-base` | `allenai/OLMo-2-0425-1B` | Base model — **no chat template**, plain-text prompt used. |
| `qwen` | `Qwen/Qwen3-1.7B` | Thinking disabled by default; `--thinking` to enable. |

> The registry in `config.py` makes adding/swapping models a one-line change.

## Key options (`generate.py`)

| flag | default | meaning |
|------|---------|---------|
| `--model` | `olmo` | Registry key. |
| `--dataset` | `V1` | `V1`, `V2`, `V3_dev`, `V3_test`. |
| `--limit` | `0` | Cap #dialogues (0 = all). |
| `--include-solution` / `--no-solution` | include | Give the tutor the reference solution as context (answer still withheld from the student). |
| `--thinking` | off | Qwen3 `<think>` traces. |
| `--temperature` / `--top-p` / `--max-tokens` | 0.7 / 0.9 / 256 | Sampling. |
| `--tp` | 1 | `tensor_parallel_size` (#GPUs). |
| `--gpu-mem-util` | 0.90 | vLLM `gpu_memory_utilization`. |
| `--enforce-eager` | off | Disable CUDA graphs if capture OOMs. |

## Scoring — LLM-as-a-judge

`score.py` rates each generated response on the 8 MRBench dimensions using a
judge model (default `openai-group/gpt-5.6-sol`) via the TrueFoundry PromptLens
gateway. This is the benchmark-native evaluation path (MRBench itself studies
Prometheus2 / Llama-3.1-8B as automatic judges).

```bash
export PROMPTLENS_API_KEY=...          # or put it in day1eval/.env
pip install requests python-dotenv     # already in requirements.txt

python score.py --self-test            # 1 tiny call — check gateway + key
python score.py --in outputs/olmo_V1.json           # score a run
python score.py --in outputs/olmo_V1.json --limit 10 --workers 8
```

- Judge config lives in `config.py` (`JUDGE_MODEL`, gateway URL, workers, retries).
  Swap to `claude-group/claude-opus-4-8` with `--model`.
- OpenAI-family judge models use `max_completion_tokens` (handled automatically);
  temperature is omitted by default (gpt-5.x reasoning models reject non-default).
- 429 / 5xx are retried with exponential backoff. The API key is read from the
  environment / `.env` and never written to output.

**Output:** `outputs/<name>_scored.json` = `{ meta, summary, records[] }`, where
each record gains a `judgment` (per-dimension labels) and `summary` holds
per-dimension label distributions + mean scores (0..1, higher = better).

### The 8 dimensions

| Dimension | Labels | "Good" = |
|-----------|--------|----------|
| Mistake_Identification | Yes / To some extent / No | Yes |
| Mistake_Location | Yes / To some extent / No | Yes |
| Revealing_of_the_Answer | Yes (correct) / Yes (incorrect) / No | **No** |
| Providing_Guidance | Yes / To some extent / No | Yes |
| Actionability | Yes / To some extent / No | Yes |
| Coherence | Yes / To some extent / No | Yes |
| Tutor_Tone | Encouraging / Neutral / Offensive | Encouraging |
| Humanlikeness | Yes / To some extent / No | Yes |

> Because these are new (generated) responses, there's no human gold label to
> compare against — the judge *is* the scorer. To calibrate the judge, you can
> point it at MRBench's original tutor responses (which have human labels) and
> measure agreement; not wired up yet.
