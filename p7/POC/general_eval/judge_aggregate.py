"""Aggregate MT-Bench-style judgments into base vs SFT.

Reads judge_out_*.json (records keyed by task_id) + judge_key.json + the results file
(for length reporting). Reports:
  - Single-answer grading: mean 1-10 per model (+ per category).
  - Pairwise win-rate with position-swap: SFT vs base (win only if consistent in BOTH orders).
"""
import json, glob, statistics as st

key = json.load(open("judge_key.json"))
recs = {r["id"]: r for r in (json.loads(l) for l in open("general_eval_results.jsonl"))}

out = {}
for f in sorted(glob.glob("judge_out_*.json")):
    for r in json.load(open(f)):
        out[r["task_id"]] = r

# ---------- single-answer grading ----------
single = {"base": [], "sft": []}
single_cat = {}
for tid, meta in key.items():
    if meta["type"] != "single":
        continue
    r = out.get(tid)
    if not r or r.get("rating") is None:
        continue
    m = meta["model"]; rating = float(r["rating"])
    single[m].append(rating)
    cat = recs[meta["prompt_id"]]["category"]
    single_cat.setdefault(cat, {"base": [], "sft": []})[m].append(rating)

# ---------- pairwise with position-swap ----------
# collect the two orderings per prompt; a model "wins" a single ordering if judged better.
per_prompt = {}
for tid, meta in key.items():
    if meta["type"] != "pairwise":
        continue
    r = out.get(tid)
    if not r:
        continue
    v = str(r.get("verdict", "")).strip().lower()
    if v in ("a", "b"):
        winner = meta["a_model"] if v == "a" else meta["b_model"]
    else:
        winner = "tie"
    per_prompt.setdefault(meta["prompt_id"], []).append(winner)

wins = {"base": 0, "sft": 0}
ties = 0
n_pairs = 0
for pid, verdicts in per_prompt.items():
    if len(verdicts) < 2:
        continue
    n_pairs += 1
    # conservative MT-Bench: win only if the SAME model wins in both orders; else tie
    if verdicts[0] == verdicts[1] and verdicts[0] in ("base", "sft"):
        wins[verdicts[0]] += 1
    else:
        ties += 1

def mean(x):
    return st.mean(x) if x else float("nan")

print("=" * 60)
print("GENERAL-INSTRUCTION EVAL  —  base vs SFT  (MT-Bench protocol)")
print("=" * 60)
print(f"\n[Single-answer grading] mean score 1-10 (higher=better), n={len(single['base'])}/model")
print(f"  base: {mean(single['base']):.2f}")
print(f"  sft : {mean(single['sft']):.2f}")
print(f"  delta (sft - base): {mean(single['sft']) - mean(single['base']):+.2f}")

print(f"\n[Pairwise win-rate, position-swap controlled] n={n_pairs} prompts")
print(f"  SFT wins : {wins['sft']:>2}  ({100*wins['sft']/n_pairs:.0f}%)")
print(f"  base wins: {wins['base']:>2}  ({100*wins['base']/n_pairs:.0f}%)")
print(f"  ties     : {ties:>2}  ({100*ties/n_pairs:.0f}%)   (incl. position-flip inconsistencies)")
sft_wr = (wins["sft"] + 0.5 * ties) / n_pairs
print(f"  SFT win-rate vs base (ties=0.5): {sft_wr:.3f}   (0.50 = parity)")

print(f"\n[Length] mean words:  base {mean([len(recs[p]['outputs']['base'].split()) for p in recs]):.0f}"
      f"  |  sft {mean([len(recs[p]['outputs']['sft'].split()) for p in recs]):.0f}")

print("\n[Single-answer by category]  base -> sft")
for cat in sorted(single_cat):
    b, s = single_cat[cat]["base"], single_cat[cat]["sft"]
    print(f"  {cat:<14} {mean(b):.1f} -> {mean(s):.1f}")

json.dump({"single_base": mean(single["base"]), "single_sft": mean(single["sft"]),
           "pairwise": {"sft_wins": wins["sft"], "base_wins": wins["base"], "ties": ties,
                        "n": n_pairs, "sft_win_rate": sft_wr}},
          open("judge_summary.json", "w"), indent=2)
print("\nwrote judge_summary.json")
