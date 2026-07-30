"""Build MT-Bench-style judging tasks from general_eval_results.jsonl.

Produces two task types, mixed + shuffled into N batches:
  - "single"  : MT-Bench single-answer grading (rate ONE response 1-10).  2 per prompt.
  - "pairwise": MT-Bench pairwise, run in BOTH orders (position-swap bias control). 2 per prompt.
Keys (model identities + order) are kept separate so judges are blind.

Usage: python judge_build.py [results_path] [n_batches]
"""
import json, random, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "general_eval_results.jsonl"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
random.seed(20)

recs = [json.loads(l) for l in open(SRC)]
tasks, key = [], {}

for r in recs:
    pid, prompt = r["id"], r["prompt"]
    base, sft = r["outputs"]["base"], r["outputs"]["sft"]

    # single-answer grading: one task per response (blind)
    for model, ans in [("base", base), ("sft", sft)]:
        tid = f"{pid}__single_{model}"
        tasks.append({"task_id": tid, "type": "single", "prompt": prompt, "answer": ans})
        key[tid] = {"type": "single", "prompt_id": pid, "model": model}

    # pairwise, both orders (position swap)
    for k, (ma, mb) in enumerate([("base", "sft"), ("sft", "base")]):
        tid = f"{pid}__pw{k}"
        aa = base if ma == "base" else sft
        bb = base if mb == "base" else sft
        tasks.append({"task_id": tid, "type": "pairwise", "prompt": prompt, "answer_a": aa, "answer_b": bb})
        key[tid] = {"type": "pairwise", "prompt_id": pid, "a_model": ma, "b_model": mb}

random.shuffle(tasks)
batches = [[] for _ in range(N)]
for i, t in enumerate(tasks):
    batches[i % N].append(t)
for k, b in enumerate(batches):
    json.dump(b, open(f"judge_batch_{k}.json", "w"), indent=2, ensure_ascii=False)
json.dump(key, open("judge_key.json", "w"), indent=2)

nS = sum(1 for v in key.values() if v["type"] == "single")
nP = sum(1 for v in key.values() if v["type"] == "pairwise")
print(f"tasks: {len(tasks)} ({nS} single + {nP} pairwise) | batches: {[len(b) for b in batches]}")
