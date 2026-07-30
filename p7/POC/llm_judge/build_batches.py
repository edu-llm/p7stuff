import json, random, os

random.seed(7)
SRC = "../test_results_instruct.jsonl"
OUT = "."
SETUPS = ["A_raw_noSI", "B_raw_SI", "C_sft_noSI", "D_sft_SI"]
N_BATCHES = 4

recs = [json.loads(l) for l in open(SRC)]
items, key = [], {}
for r in recs:
    did = r["dialogue_id"]
    order = SETUPS[:]
    random.shuffle(order)                       # blind: randomize which candidate is which setup
    cands = []
    for j, s in enumerate(order):
        rid = f"{did}__R{j+1}"
        cands.append({"rid": rid, "response": r["outputs"][s]})
        key[rid] = s
    items.append({
        "problem_id": did,
        "problem": r["problem"],
        "final_answer": r.get("answer"),   # numeric answer only (to detect reveal/correctness); NOT the gold tutor turn
        "candidates": cands,
    })

# split into N_BATCHES
batches = [[] for _ in range(N_BATCHES)]
for i, it in enumerate(items):
    batches[i % N_BATCHES].append(it)

for k, b in enumerate(batches):
    json.dump(b, open(os.path.join(OUT, f"judge_batch_{k}.json"), "w"), indent=2, ensure_ascii=False)
json.dump(key, open(os.path.join(OUT, "judge_key.json"), "w"), indent=2)
print("items:", len(items), "| batches:", [len(b) for b in batches], "| candidates/item: 4")
