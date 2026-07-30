"""Builds math_logic_eval_colab.ipynb from math_logic_prompts.jsonl (no nbformat dep)."""
import json

prompts = [json.loads(l) for l in open("math_logic_prompts.jsonl")]
# strip gold from what goes in the notebook (model must never see it); keep id/meta only
pub = [{"id": r["id"], "source": r["source"], "category": r["category"],
        "difficulty": r["difficulty"], "answer_type": r["answer_type"],
        "prompt": r["prompt"]} for r in prompts]
prompts_literal = "PROMPTS = " + json.dumps(pub, ensure_ascii=False, indent=0)

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}

cells = []
cells.append(md(
"""# Math + Logic eval — base Instruct vs your SFT model (frontier benchmarks)

Runs **`allenai/OLMo-2-0425-1B-Instruct`** and your **SFT LoRA adapter** on **70 real items** drawn
from frontier benchmarks — **GSM8K**, **MATH-500 (level 5)**, **BBH logical-deduction**, and
**AIME 2024** — using **chain-of-thought + greedy decoding** (the standard protocol for these).

Pick where the "give the answer" directive lives with **`ARM`**: `"nosi"` (none), `"directsi"`
(a system prompt), or `"userinstr"` (inside the question). Same treatment for both models. Saves
`math_logic_results_<ARM>.jsonl`. **Grading is verifiable final-answer accuracy**, done offline
(`grade_math_logic.py`) — no API key.

Run-all: pick a GPU runtime, set `SFT_MODEL` (defaults to the Drive path), Run All. Only the Drive
auth popup is interactive."""))

cells.append(code(
"""# 1. Install
!pip -q install -U transformers accelerate peft safetensors
!pip -q uninstall -y torchao   # peft rejects Colab's old torchao 0.10; we don't use it
import torch, transformers
print("transformers", transformers.__version__, "| cuda:", torch.cuda.is_available())"""))

cells.append(code(
"""# 2. Config
BASE_MODEL = "allenai/OLMo-2-0425-1B-Instruct"
# Your SFT LoRA adapter (folder with adapter_config.json):
SFT_MODEL = "/content/drive/MyDrive/olmo2_socratic_sft/instruct/olmo2-1b-socratic-tutor-instruct/checkpoint-923"

MOUNT_DRIVE = True
GREEDY = True               # frontier math/logic protocol = greedy pass@1
GEN_MAX_NEW = 1024          # generous: avoid truncation before \\boxed{}
BATCH_SIZE = 8
SEED = 0

# ARM controls WHERE the "give the answer" directive lives (both models get identical treatment):
#   "nosi"      : no system prompt; only a minimal "put answer in \\boxed{}" hint in the question.
#   "directsi"  : the directive is a SYSTEM prompt (DIRECT_SI). NOTE: the SFT model learned that the
#                 mere presence of a system message == "be Socratic" (100% of pedagogy training data
#                 had a system prompt, 0% of general data did), so this arm triggers hand-back.
#   "userinstr" : the directive is put INSIDE the user question instead ("...  Solve it and give the
#                 final answer.") — the in-distribution channel — to avoid the system-prompt shortcut.
ARM = "userinstr"
DIRECT_SI = "You are a math and logic problem solver. Solve the problem and give the final answer inside \\\\boxed{ }."
RESULTS_PATH = f"math_logic_results_{ARM}.jsonl"

if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")"""))

cells.append(code("# 3. Prompt bank (70 frontier items; gold answers withheld from the model)\n" + prompts_literal +
"\nprint(len(PROMPTS), 'items |', sorted(set(p['source'] for p in PROMPTS)))"))

cells.append(code(
'''# 4. Loader + CoT prompt builder + generation (greedy)
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftConfig, PeftModel
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16

def _is_adapter(mid):
    try:
        PeftConfig.from_pretrained(mid); return True
    except Exception:
        return False

def load_model(model_id, base_fallback=BASE_MODEL):
    is_ad = _is_adapter(model_id)
    base_id = PeftConfig.from_pretrained(model_id).base_model_name_or_path or base_fallback if is_ad else base_fallback
    print(f"loading {model_id}" + (f"  (LoRA on {base_id})" if is_ad else ""))
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(base_id if is_ad else model_id, dtype=DTYPE, trust_remote_code=True)
    if is_ad:
        m = PeftModel.from_pretrained(m, model_id).merge_and_unload()
    return m.to(DEVICE).eval(), tok

def build_user_prompt(p):
    q = p["prompt"]
    box = ("Put ONLY the letter of the correct option inside \\\\boxed{ }, e.g. \\\\boxed{C}."
           if p["answer_type"] == "mc" else "Put your final answer inside \\\\boxed{ }.")
    # userinstr arm: the "give the answer" directive rides IN the question (user channel),
    # not as a system prompt, so it does not fire the SFT system-message->Socratic shortcut.
    lead = "Solve it and give the final answer. " if ARM == "userinstr" else ""
    return q + "\\n\\n" + lead + box

@torch.no_grad()
def generate_all(model, tok, prompts):
    set_seed(SEED)
    msgs = []
    for p in prompts:
        conv = [{"role": "system", "content": DIRECT_SI}] if ARM == "directsi" else []
        conv.append({"role": "user", "content": build_user_prompt(p)})
        msgs.append(tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True))
    out = []
    for b in range(0, len(msgs), BATCH_SIZE):
        enc = tok(msgs[b:b+BATCH_SIZE], return_tensors="pt", padding=True, truncation=True,
                  max_length=2048, add_special_tokens=False).to(DEVICE)
        kw = dict(max_new_tokens=GEN_MAX_NEW, pad_token_id=tok.pad_token_id)
        kw.update(dict(do_sample=False) if GREEDY else dict(do_sample=True, temperature=0.7, top_p=0.9))
        gen = model.generate(**enc, **kw)
        new = gen[:, enc["input_ids"].shape[1]:]
        out.extend(t.strip() for t in tok.batch_decode(new, skip_special_tokens=True))
        print(f"  {min(b+BATCH_SIZE, len(msgs))}/{len(msgs)}")
    return out'''))

cells.append(code(
'''# 5. Generate for BOTH models
import gc
results = [{"id": p["id"], "source": p["source"], "category": p["category"],
           "difficulty": p["difficulty"], "answer_type": p["answer_type"],
           "prompt": p["prompt"], "outputs": {}} for p in PROMPTS]

for label, mid in [("base", BASE_MODEL), ("sft", SFT_MODEL)]:
    model, tok = load_model(mid)
    print(f"generating {label} ...")
    outs = generate_all(model, tok, PROMPTS)
    for r, o in zip(results, outs):
        r["outputs"][label] = o
    del model; gc.collect(); torch.cuda.empty_cache()

print("done. sample:")
print(results[0]["prompt"][:120])
print("  base:", results[0]["outputs"]["base"][:160])
print("  sft :", results[0]["outputs"]["sft"][:160])'''))

cells.append(code(
'''# 6. Save results (+ Drive backup)
import json, os, shutil
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\\n")
print("wrote", RESULTS_PATH, "with", len(results), "records")

if MOUNT_DRIVE:
    dst = "/content/drive/MyDrive/olmo2_socratic_sft/instruct"
    os.makedirs(dst, exist_ok=True)
    shutil.copy(RESULTS_PATH, dst)
    print("backed up ->", os.path.join(dst, RESULTS_PATH))

# Download math_logic_results.jsonl (left file panel) and send it back for offline grading.'''))

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

json.dump(nb, open("math_logic_eval_colab.ipynb", "w"), ensure_ascii=False, indent=1)
print("wrote math_logic_eval_colab.ipynb with", len(cells), "cells")
