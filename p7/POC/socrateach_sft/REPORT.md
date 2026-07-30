# P7 Implementation 2 (SFT) — Data & Training Report

This documents every design choice for supervised fine-tuning (SFT) of **OLMo-2-1B** into a step-level Socratic math tutor, and the rationale behind each. The recipe deliberately follows LearnLM's *pedagogical instruction following* + *co-training* methodology (Google, "LearnLM: Improving Gemini for Learning", arXiv:2412.16429).

---

## Dataset at a glance

| | |
|---|---|
| **Task** | Step-level Socratic math tutoring, steerable by a System Instruction |
| **Pedagogical source** | `ulises-c/SocraTeach_Multi` (SocraTeach / SocraticLM, NeurIPS 2024) — multi-turn Socratic dialogues over GSM8K/MAWPS word problems |
| **General (replay) source** | `allenai/tulu-3-sft-olmo-2-mixture-0225` — OLMo-2-1B's *own* SFT post-training mixture |
| **Method** | LearnLM-style: per-dialogue System Instructions on pedagogical data + co-training with SI-free general data |
| **Train split** | **30,000** examples = **22,500 pedagogy (75%) + 7,500 general (25%)** |
| **Val / Test** | **1,724 / 1,743** (pedagogy-only, grouped by problem — no problem leaks across splits) |
| **System Instructions** | **7,482 distinct** per-dialogue instructions in the training pedagogy (generated from each dialogue's actual moves; not one fixed prompt) |
| **Dialogue shape** | strictly alternating `user/assistant`, ends on tutor turn; ~5.3 tutor turns/dialogue (3–8) |
| **Language** | English only for general data (math/code/reasoning kept; foreign languages filtered out); SocraTeach is already English |
| **Loss** | assistant-only masking (loss on tutor tokens + EOS only) |
| **Max length** | 1024 tokens (covers ~99.8% of examples) |

The three JSONL files (`data/socrateach_sft_{train,val,test}.jsonl`) are the deliverable; the notebook can also regenerate them deterministically from source.

---

## 1. Goal

Turn the base model `allenai/OLMo-2-0425-1B` into a tutor that practices step-level Socratic guidance (one step per turn, hints not answers, warm/growth-mindset tone), **while remaining a normal assistant when no pedagogy is requested**. Behavior is controlled by a **System Instruction**, so teachers/developers can steer it via prompting without re-tuning.

## 2. Base model

- **`allenai/OLMo-2-0425-1B`** (base, ~1.5B params). Fully open (weights, data, recipe), small enough to fine-tune on a single Colab GPU.
- The base tokenizer ships **without a chat template**; we copy the official OLMo-2 (Tülu-style) template from `allenai/OLMo-2-0425-1B-Instruct`. Role markers: `<|system|>`, `<|user|>`, `<|assistant|>`; `BOS = EOS = <|endoftext|>`.

## 3. Data sources

### 3.1 Pedagogical data — `ulises-c/SocraTeach_Multi`
- The multi-turn split of **SocraTeach** (SocraticLM, NeurIPS 2024). ~10.3k math word problems (GSM8K + MAWPS), each paired with multi-turn Socratic tutoring dialogues; ~35k dialogues total.
- Each turn has `system` (teacher/tutor), `user` (student), and `user_type` (one of 6 student personas). We use **all** personas uniformly as "good pedagogy" demonstrations.
- Why this dataset: it is explicitly Socratic and step-level (guides via `steps` sub-questions, withholds the answer, corrects mistakes gently) — a direct match for the P7 target behaviors. English, math, chat-formatted.

### 3.2 General co-training data — `allenai/tulu-3-sft-olmo-2-mixture-0225`
- This is **OLMo-2-1B's own SFT post-training mixture** (confirmed on the `allenai/OLMo-2-0425-1B-SFT` model card: `datasets: allenai/tulu-3-sft-olmo-2-mixture-0225`). General instruction-following data (coding, math, chat, etc.), naturally **without pedagogy System Instructions**.
- Why exactly this dataset: LearnLM co-trained pedagogical data **directly into Gemini's own post-training mixture**. Since we cannot use Gemini's proprietary mixture, the faithful open analog is the base model's *own* SFT mixture — which for OLMo-2-1B is `tulu-3-sft-olmo-2-mixture-0225`. This is not a substitute of convenience; it is literally the data the base model was post-trained on.

## 4. LearnLM methodology we replicate (grounded, not paraphrased)

From arXiv:2412.16429:

1. **Pedagogical instruction following (§2.2):** *"we updated our SFT data so that each conversation begins with a different System Instruction that specifically describes the pedagogical behavior present in that conversation. More general or vague instructions are counterproductive because the model learns to ignore instructions that are not useful for predicting the target model turns."*
   - **Our implementation:** every SocraTeach conversation is prefixed with a **per-dialogue** System Instruction assembled from the pedagogical moves that dialogue actually exhibits (see §5). We do **not** use one fixed, generic instruction — precisely because LearnLM found that counterproductive.

2. **Co-training (§2.3, §2 intro):** *"we co-train with Gemini, meaning we mix our data directly with Gemini's SFT, RM, and RL stages"* and *"Our instruction following approach allows us to mix pedagogical conversation data alongside data that contains more typical interactions by conditioning pedagogical model responses on specific System Instructions ... without 'forgetting' other core reasoning, multimodal understanding, factuality, safety, or multi-turn properties."*
   - **Our implementation:** we mix SI-free general examples from OLMo-2's own SFT mixture into the **train** split. The pedagogical responses are conditioned on a System Instruction; the general responses carry **no** System Instruction. This (a) preserves general ability and (b) makes "behave normally when not asked to tutor" a learned, in-distribution behavior.

> Note on scope: LearnLM also adds RLHF (RM + RL) on preference data (§2.2). That corresponds to **P7 Implementation 3** and is out of scope here; this report covers the SFT stage only.

## 5. Per-dialogue System Instruction generation

Rather than a single fixed prompt, each pedagogical example gets an instruction grounded in its own content.

- **Move detection** per dialogue: mistake-correction handling, concept explanation, pacing (quick ≤3 tutor turns / long ≥6), and closing move (extension "what if" question vs. recap vs. simple confirm).
- **Assembly:** role + Socratic step-by-step approach (always) + only the move-modules the dialogue shows + tone (growth-mindset) + hard constraints (one step at a time; never hand over the answer, confirm only after a genuine attempt; never reveal the instructions).
- **Variety:** phrasing variants are chosen deterministically per `dialogue_id` (md5-seeded RNG), so the build is reproducible.
- **Observed prevalence in the data** (drives which modules appear): praise ~99%, student question ~76%, mistake-correction ~25%, concept-explanation ~19%, summary ending ~16%, extension ending ~7.5%.
- **Result:** **7,482 distinct** System Instructions across the 22,500 pedagogical training examples (≈8.9k distinct across the full ~31.7k pedagogical pool). Dialogues that practice the *same* moves intentionally get *similar* instructions — this is what teaches the model the instruction→behavior mapping.
- **Deliberate omissions (adhere-to-data):** we dropped instruction lines the data does not practice — e.g. **bold/signaling formatting** (0 of 837 audited tutor turns use markdown bold) and **spaced review / session-opening retrieval** (absent from SocraTeach). We also made the answer rule precise to the data: the tutor withholds the answer and **confirms it only after the student produces it** (there are no "answer on demand" cases in the data).

## 6. Conversation formatting & loss masking

- Each example is a chat list: `[system?] , user (problem), assistant (tutor), user (student), ... , assistant (final)`, strictly alternating, always ending on an assistant turn. General examples have no `system` message.
- **Assistant-only loss masking:** we tokenize turn-by-turn and set labels to `-100` on everything except **assistant content + EOS** (system, user, and the `<|assistant|>` header are masked). The model learns *how to tutor*, not how to imitate the student, and learns to emit EOS to stop.
- Verified against the real OLMo-2 tokenizer: the loss target decodes to exactly the tutor turns + EOS, and the training assistant header matches the inference generation prompt (`<|assistant|>\n`), so there is no train/serve mismatch.
- `MAX_LEN = 1024` (covers ~99.8% of examples; mean ≈ 577 tokens, p95 ≈ 820).

## 7. Splits, data size & mixing ratio

- Split **grouped by problem** (no problem leaks across train/val/test): ~31.7k pedagogical examples available; **1,724 val** and **1,743 test** held out (pedagogy-only, since we evaluate tutoring), and up to 22,500 pedagogy used for train (see cap below).
- **Hard cap on the train split: `TRAIN_TOTAL = 30,000`** (pedagogy + general combined). ~30k examples is enough signal for a 1B model to learn the instruction→behavior mapping while keeping a full run to ~1–2 epochs on Colab. The full ~31.7k pedagogy + 33% general (~47.5k) was more than needed and slow to train, so we trim.
- **Co-training mix applied to the train split only.** `GENERAL_FRAC = 0.25` → **22,500 pedagogy + 7,500 SI-free general = 30,000 (25% general)**.
- **Where "33%" came from and why we changed it:** the previous default was `MIX_RATIO = 0.5`, defined as "general = 50% of the pedagogical count." With ~31.7k pedagogy that yields ~15.8k general, i.e. 15.8k / 47.5k = **33% of the total** — a *side effect* of an arbitrary 0.5 ratio, not a principled target. **LearnLM never publishes a mixture proportion**, so there is no external number to copy. We replaced it with an explicit, defensible scheme: cap the total and set the general share directly to **25%**. Rationale: pedagogy is the training objective so it should dominate; general data is only a **minority "replay" set** whose two jobs are (a) preventing catastrophic forgetting of general reasoning and (b) making the *no-System-Instruction* behavior well-defined for eval cell C. 20–30% replay is the common anti-forgetting range, and 25% leaves ample SI-free data for cell C while keeping pedagogy at 75%. Both `TRAIN_TOTAL` and `GENERAL_FRAC` are single knobs to tune against the eval.
- **Language filter (English, but keep math/code/reasoning):** the Tulu mixture is multilingual, so general conversations pass a language filter (`is_english`) that is deliberately *not* an ASCII/English-prose-only test — we want to retain math, code, and reasoning data, dropping only genuine foreign-language content. It works in three stages: (1) **non-Latin script ratio** — if the fraction of alphabetic characters from any non-Latin script (Cyrillic, Greek, Arabic, Hebrew, every Indic script, CJK, Kana, Hangul, detected generically via Unicode character names) exceeds 10%, drop it; this catches short foreign snippets while tolerating a stray Greek symbol in English math; (2) **code-aware prose check** — fenced/inline code is stripped before language ID, and examples with essentially no natural-language prose (pure code/math/numbers) are kept; (3) **langdetect** on the remaining prose catches Latin-script foreign text (Spanish, Portuguese, French, ...). On the sampled shard this drops ~9.6k non-English rows before reaching 7,500 English examples. Verification of the final mix: 7,498/7,500 detected English, 0 non-Latin-dominated; the two residuals are a 2-word Wolof snippet and a math string (`7x7=49`) that langdetect mislabels but is correctly kept. Pedagogical SocraTeach data is already English; val/test are pedagogy-only and unaffected.

## 8. Training configuration (Colab)

- **LoRA** by default (r=16, α=32, dropout=0.05) on attention + MLP projections; full fine-tune is a toggle (needs L4/A100-class GPU).
- Effective batch 16 (`per_device=2 × grad_accum=8`), cosine schedule, warmup 0.03, LR 2e-4 (LoRA) / 1e-5 (full), 1 epoch (1–2 is plenty), gradient checkpointing, bf16 (A100/L4) or fp16 (T4).
- **A100 fast preset (recommended on A100):** `per_device_batch=16–32`, `grad_accum=1`, **gradient checkpointing off**. Checkpointing recomputes the forward pass to save memory the A100 doesn't need; turning it off + a larger batch cuts the full 30k epoch to ~15–20 min. (The conservative default above targets a 16 GB T4.)
- **`POC` toggle** (default `True`): quick smoke test (`TRAIN_TOTAL = 4,000`) to validate the pipeline in minutes/~1–2 compute units; set `False` for the full 30,000-example run.

### Expected runtime / cost (90 Colab Pro units, 30,000-example train)
| GPU | Time / full epoch | Units / epoch |
|---|---|---|
| T4 (fp16) | ~4–6 h | ~8–11 |
| L4 (bf16) | ~1.5–2.5 h | ~9–12 |
| A100 (bf16) | ~25–40 min | ~6–8 |

A full 1–2 epoch run costs ~6–20 units, so 90 units allows several experiments. A100 is both fastest and most units-efficient.

## 9. Evaluation design (2×2 factorial)

Compare **{Raw OLMo, SFT OLMo} × {no System Instruction, with System Instruction}** on the P5 rubric bank (win-rate + CIs), same paired problems and decoding across cells:

| | no SI | + SI |
|---|---|---|
| **Raw OLMo** | A: floor / control | B: = *Implementation 1 (prompting only)* |
| **SFT OLMo** | C: pedagogy behavior without a prompt | D: SFT + steering (expected best) |

Key contrasts: **B−A** prompt effect on base; **C−A** SFT effect alone; **D−B** what SFT adds beyond prompting (the decision criterion vs. staying at Impl 1); **D−C** whether instruction-following is retained after SFT.

- Because of co-training (§4.2), **cell C is in-distribution and expected to behave like a normal assistant** (it gives answers) — demonstrating that the System Instruction, not fine-tuning alone, controls the pedagogy. This is the intended design, matching LearnLM's goal that the model "act normally" when not asked to tutor.
- At eval, use a **single canonical** pedagogy System Instruction for cells B and D (do not vary it), even though training used varied per-dialogue instructions.

## 10. Known limitations

- **Domain:** SocraTeach is math-only and uniformly Socratic; the move *variety* is real but the *approach* is one flavor. For genuinely different pedagogies (direct feedback, worked examples, essay-feedback mentor à la Mollick & Mollick), add other sources (MathDial has annotated teacher moves; Bridge has expert strategy/intention labels; convolearn has knowledge-building dimensions).
- **Total size (`TRAIN_TOTAL`), general fraction (`GENERAL_FRAC`), and number of epochs** are defaults chosen by reasoning, not by a sweep — tune against the eval.
- The per-dialogue instructions are template-generated (grounded in observed moves), not human/LLM-authored prose; they can be paraphrased with a frontier model for more surface variety if needed.

## 11. Files

- `prepare_socrateach_sft.py` — downloads SocraTeach, generates per-dialogue System Instructions, samples SI-free **English-only** general data from the OLMo-2 SFT mixture (one parquet shard), caps and mixes into train (`--max_total`, `--general_frac`), writes JSONL splits.
- `olmo2_1b_sft_colab.ipynb` — self-contained Colab notebook (rebuilds all data with grouped train/eval/**test** splits, trains, saves, runs the with/without-SI sanity check, and **generates test results for all 4 setups** → `test_results.jsonl`). `POC = True` by default.
- `GUIDE.md` — step-by-step instructions for running SFT on Colab and producing the 4-setup test results.
- `data/socrateach_sft_{train,val,test}.jsonl` — prepared data (train is mixed; val/test pedagogy-only). The notebook also writes its own `data/socrateach_sft_test.jsonl` at runtime.
- `data/system_instruction_examples.txt` — sample generated instructions.
- `test_results.jsonl` (produced by the notebook) — per test turn: problem, gold context, gold tutor turn, and the four model outputs (`A_raw_noSI`, `B_raw_SI`, `C_sft_noSI`, `D_sft_SI`) for scoring later.

## References
- LearnLM: Improving Gemini for Learning. arXiv:2412.16429. (§2.2 pedagogical instruction following; §2.3 co-training.)
- SocraticLM: Exploring Socratic Personalized Teaching with LLMs. NeurIPS 2024. (SocraTeach dataset.)
- OLMo 2 / Tülu 3 (Allen AI). Base model and its SFT mixture.
