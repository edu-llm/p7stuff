"""Split a needs_verify_*.json into N blind LLM-as-verifier batches for subagents.

Each task: {task_id, question, gold, candidate}. The verifier decides whether the candidate
response's FINAL answer is mathematically equivalent to gold -> "correct"/"incorrect".
Model identity is not revealed.

Usage: python build_verify_batches.py [needs_verify_TAG.json] [n_batches]
Batches are written as verify_TAG_batch_*.json; verifiers should write verifier_out_TAG_*.json.
"""
import json, sys, re

_pos = [a for a in sys.argv[1:] if not a.startswith("--")]
NEEDS = _pos[0] if _pos else "needs_verify_nosi.json"
N = int(_pos[1]) if len(_pos) > 1 else 4
m = re.search(r"needs_verify_(\w+)\.json", NEEDS)
TAG = m.group(1) if m else "nosi"

tasks = json.load(open(NEEDS))
pub = [{"task_id": t["task_id"], "question": t["question"], "gold": t["gold"],
        "candidate": t["candidate"]} for t in tasks]
batches = [[] for _ in range(N)]
for i, t in enumerate(pub):
    batches[i % N].append(t)
for k, b in enumerate(batches):
    json.dump(b, open(f"verify_{TAG}_batch_{k}.json", "w"), indent=2, ensure_ascii=False)
print(f"{len(pub)} verify tasks (tag={TAG}) -> {N} batches: {[len(b) for b in batches]}")
print(f"verifiers read verify_{TAG}_batch_k.json -> write verifier_out_{TAG}_k.json")
