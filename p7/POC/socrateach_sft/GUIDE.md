# Step-by-step: SFT OLMo-2-1B and generate test results (4 setups)

This guide walks you through fine-tuning `allenai/OLMo-2-0425-1B` into a Socratic
math tutor with `olmo2_1b_sft_colab.ipynb`, and producing model outputs on the
held-out test set for the 2×2 evaluation design. **Scoring/evals are run later**;
here you only train and generate the outputs.

By default the notebook uses **your prepared JSONL files** (you upload them, or point
it at Drive) — nothing is regenerated. If you'd rather have it rebuild the data from
Hugging Face instead, set `LOAD_FROM_FILES = False` in Cell 4.

---

## 0. What you'll produce

- `olmo2-1b-socratic-tutor/` — the fine-tuned model (LoRA adapter by default).
- `data/socrateach_sft_test.jsonl` — the held-out test split (grouped by problem).
- `test_results.jsonl` — model outputs for all **four** setups on the test set:

| | no System Instruction | + System Instruction |
|---|---|---|
| **Raw OLMo** | **A** `A_raw_noSI` (control) | **B** `B_raw_SI` (= Implementation 1, prompting only) |
| **SFT OLMo** | **C** `C_sft_noSI` (pedagogy without a prompt) | **D** `D_sft_SI` (SFT + steering, expected best) |

---

## 1. Open the notebook on Colab

1. Upload `olmo2_1b_sft_colab.ipynb` to [Google Colab](https://colab.research.google.com/)
   (File → Upload notebook), or put it in Drive and open it.
2. **Runtime → Change runtime type → Hardware accelerator = GPU.** Pick the best
   GPU your plan offers: **A100 > L4 > T4** (A100 is fastest *and* most units-efficient).
3. (Optional) Runtime → View resources to confirm a GPU is attached.
4. Have your three data files handy to upload when prompted (Section 2b):
   `socrateach_sft_train.jsonl`, `socrateach_sft_val.jsonl`, `socrateach_sft_test.jsonl`
   (from your local `socrateach_sft/data/` folder). Alternatively, mount Drive and put
   them in `DATA_DIR` so no upload is needed.

## 2. Smoke test first (PoC), then the full run

The notebook has a single switch in **Cell 4 (Configuration)**:

- `POC = True` (default): tiny run (`TRAIN_TOTAL = 4000`), finishes in minutes and
  costs ~1–2 compute units. **Run this first** to confirm the whole pipeline works
  end to end (data → train → save → test_results).
- `POC = False`: the real run (`TRAIN_TOTAL = 30000` = 22,500 pedagogy + 7,500
  general).

Other knobs you may touch in Cell 4:

| Knob | Default | Meaning |
|---|---|---|
| `USE_LORA` | `True` | LoRA (fits T4). Set `False` for full fine-tune (needs L4/A100). |
| `GENERAL_FRAC` | `0.25` | Share of train that is SI-free English general "replay" data. |
| `TRAIN_TOTAL` | 4000 / 30000 | Hard cap on train size (pedagogy + general). |
| `N_EVAL_DIALOGUES` | `50` | How many test dialogues to generate outputs for. |
| `MAX_EVAL_TURNS` | `1` | Tutor turns generated per dialogue (raise for multi-turn). |
| `GEN_MAX_NEW` | `220` | Max new tokens per generated tutor turn. |
| `NUM_EPOCHS` | `1` | 1–2 is plenty. |

## 3. Run the cells top to bottom

Use **Runtime → Run all**, or run each cell in order. What each section does:

1. **Install dependencies** — `transformers`, `datasets`, `accelerate`, `peft`,
   `langdetect`. Prints the GPU name and bf16 support.
2. **Configuration** — sets the knobs above.
2b. **Data source** — with `LOAD_FROM_FILES = True` (default), loads your three JSONL
   files: it reads them from `DATA_DIR` if present, otherwise opens a **file picker** so
   you can upload `socrateach_sft_train/val/test.jsonl`. Prints the loaded counts
   (e.g. `train=30000 {'pedagogy':22500,'general':7500}`). If files are missing it
   falls through to rebuilding.
3. **Build the SFT dataset** — *skipped entirely when your files loaded in 2b.* Only if
   no files were provided does it rebuild from Hugging Face (downloads SocraTeach +
   general data, regenerates the per-dialogue System Instructions, and makes grouped
   splits). In both cases it ends with a pedagogy sample and a general sample preview.
   - Downloads `ulises-c/SocraTeach_Multi`, and for **each dialogue** builds a
     per-dialogue System Instruction describing the pedagogy that dialogue actually
     practices (Socratic step-by-step always; mistake-handling / explanation /
     pacing / closing move only when present).
   - Groups by problem and carves out **grouped** test / eval / train splits so **no
     problem leaks** across splits.
   - Mixes in SI-free, **English-only** general data from
     `allenai/tulu-3-sft-olmo-2-mixture-0225` (math/code/reasoning kept; foreign
     languages dropped) — LearnLM-style co-training.
   - Writes `data/socrateach_sft_test.jsonl` and prints TRAIN/EVAL/TEST sizes plus
     a pedagogy sample and a general sample.
4. **Load OLMo-2-1B + tokenizer** — base model has no chat template, so the notebook
   copies OLMo-2's official template from the Instruct tokenizer. Wraps in LoRA.
5. **Tokenize (assistant-only loss masking)** — only tutor tokens + EOS contribute
   to the loss; system/user tokens are masked with `-100`. Prints token-length stats.
6. **Train** — `Trainer` with gradient checkpointing, cosine schedule, bf16/fp16.
   Watch the loss go down; eval loss is logged every 200 steps.
7. **Save** — writes the model/adapter to `OUTPUT_DIR`. Set `MERGE_LORA = True` if
   you want a standalone merged model. (Optional Drive-copy snippet included.)
8. **Quick inference sanity check** — one problem, with vs. without the System
   Instruction, so you can eyeball that "+SI tutors, no-SI answers normally".
9. **Generate test results — the 4 setups** — this is the deliverable:
   - Loads a **fresh raw** base model (for A/B) alongside the **SFT** model (C/D).
   - For each of `N_EVAL_DIALOGUES` test dialogues, teacher-forces the gold student
     turns and generates the tutor's reply for the first `MAX_EVAL_TURNS` turn(s),
     under **identical problems and greedy decoding**.
   - `+SI` cells (B, D) use one fixed `CANONICAL_SI`; `no-SI` cells (A, C) use no
     system message.
   - Writes everything to `test_results.jsonl` and prints one sample across all four
     setups.

## 4. Download / save your outputs

Colab disks are ephemeral. Before the session ends, save what you need:

```python
from google.colab import files
files.download("test_results.jsonl")
```

Or copy to Drive:

```python
from google.colab import drive; drive.mount('/content/drive')
!cp test_results.jsonl /content/drive/MyDrive/
!cp -r olmo2-1b-socratic-tutor /content/drive/MyDrive/
!cp data/socrateach_sft_test.jsonl /content/drive/MyDrive/
```

---

## 5. `test_results.jsonl` format (for scoring later)

One JSON object per generated tutor turn:

```json
{
  "dialogue_id": "…",
  "turn": 0,
  "problem": "A store sells notebooks for $3 each …",
  "context": [{"role": "user", "content": "…problem…"}],
  "gold_tutor": "…the reference tutor turn from the dataset…",
  "answer": "…final answer of the problem…",
  "outputs": {
    "A_raw_noSI": "…",
    "B_raw_SI":   "…",
    "C_sft_noSI": "…",
    "D_sft_SI":   "…"
  }
}
```

- `context` is the gold conversation history shown to the model (it always ends on a
  student turn; the model generates the next tutor turn).
- `outputs` holds the four candidate tutor turns for the same context.
- `gold_tutor` is the dataset's reference turn (useful for reference-based metrics).

When you build the eval, score each `outputs[*]` with your rubric (e.g. `joe_rubric.txt`
/ the P5 rubric bank) and compare cells:

- **B − A**: prompting effect on the base model.
- **C − A**: what SFT alone buys (behavior without a prompt).
- **D − B**: what SFT adds beyond prompting (the decision criterion vs. staying at
  Implementation 1).
- **D − C**: whether instruction-following is retained after SFT.

---

## 6. Expected runtime & cost (Colab Pro, full 30k run, 1 epoch)

| GPU | Training | Units |
|---|---|---|
| T4 (fp16) | ~4–6 h | ~8–11 |
| L4 (bf16) | ~1.5–2.5 h | ~9–12 |
| A100 (bf16) | ~25–40 min | ~6–8 |

Test-result generation (50 dialogues × 1 turn × 4 setups = 200 greedy generations)
adds roughly ~5–15 min on T4, less on L4/A100. With 90 units you can comfortably run
the PoC plus a full run (and a couple of variations).

## 7. Troubleshooting

- **No GPU / `CUDA available: False`** → Runtime → Change runtime type → GPU, then
  Runtime → Restart and run all.
- **`ImportError: incompatible version of torchao ... 0.10.0`** (at `get_peft_model`) →
  Colab's pre-installed `torchao` is too old for the latest `peft`. The install cell now
  runs `pip uninstall -y torchao` (torchao is unused here). If you already hit the error,
  re-run the install cell, then **Runtime → Restart session** and Run all. Alternatively
  `!pip install -U torchao` also works but is heavier.
- **Out of memory (OOM)** → keep `USE_LORA = True`; lower `PER_DEVICE_BATCH` to 1 and
  raise `GRAD_ACCUM` to 16; reduce `MAX_LEN` (e.g. 768). For the test-results cell,
  lower `N_EVAL_DIALOGUES` or `GEN_MAX_NEW`.
- **First run is slow** → the dataset downloads (SocraTeach + one general parquet
  shard) are cached after the first time.
- **Want a standalone model** (not adapter) → set `MERGE_LORA = True` in the Save cell.
- **Reproducibility** → test generation uses greedy decoding (`do_sample=False`), so
  the four setups differ only by model and System Instruction.
