"""Build a math+logic eval set from FRONTIER benchmarks (no self-authored problems).

Sources (all with verifiable gold answers -> standard final-answer accuracy rubric):
  - GSM8K            grade-school math          (openai/gsm8k)            [anchor]
  - MATH-500         competition math, hard     (HuggingFaceH4/MATH-500)  [OpenAI/o1 subset]
  - BBH logical-ded. logic / deduction          (lukaemon/bbh)
  - AIME 2024        olympiad, very hard         (Maxwell-Jia/AIME_2024)

Output: math_logic_prompts.jsonl with fields
  id, source, category(math|logic), difficulty, prompt, gold, answer_type(int|expr|mc)
"""
import json, random, re
from datasets import load_dataset

random.seed(7)
out = []

# ---- GSM8K (integer answers) ----
gsm = load_dataset("openai/gsm8k", "main", split="test")
idx = random.sample(range(len(gsm)), 15)
for i in idx:
    q = gsm[i]["question"]
    a = gsm[i]["answer"].split("####")[-1].strip().replace(",", "")
    out.append({"id": f"gsm8k_{i}", "source": "GSM8K", "category": "math",
                "difficulty": "easy", "prompt": q, "gold": a, "answer_type": "int"})

# ---- MATH-500 level 5 (hard competition math) ----
math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
hard = [r for r in math500 if int(r.get("level", 0)) == 5]
random.shuffle(hard)
for r in hard[:25]:
    out.append({"id": f"math500_{r['unique_id'].strip('/').replace('/','_')}",
                "source": "MATH-500", "category": "math", "difficulty": "hard",
                "prompt": r["problem"], "gold": r["answer"], "answer_type": "expr"})

# ---- BBH logical deduction (logic, multiple choice) ----
bbh = load_dataset("lukaemon/bbh", "logical_deduction_seven_objects", split="test")
idx = random.sample(range(len(bbh)), 15)
for i in idx:
    out.append({"id": f"bbh_ld7_{i}", "source": "BBH-logical_deduction",
                "category": "logic", "difficulty": "hard",
                "prompt": bbh[i]["input"], "gold": bbh[i]["target"], "answer_type": "mc"})

# ---- AIME 2024 (integer 0-999, very hard) ----
aime = load_dataset("Maxwell-Jia/AIME_2024", split="train")
for r in list(aime)[:15]:
    out.append({"id": f"aime2024_{r['ID']}", "source": "AIME-2024", "category": "math",
                "difficulty": "very_hard", "prompt": r["Problem"],
                "gold": str(r["Answer"]).strip(), "answer_type": "int"})

with open("math_logic_prompts.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

from collections import Counter
print("total:", len(out))
print("by source:", dict(Counter(r["source"] for r in out)))
print("by category:", dict(Counter(r["category"] for r in out)))
