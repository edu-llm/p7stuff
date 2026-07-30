import json, glob, statistics as st

key = json.load(open("judge_key.json"))
SETUPS = ["A_raw_noSI", "B_raw_SI", "C_sft_noSI", "D_sft_SI"]

# Dimensions + label->score maps (MRBench verbatim + Joe's two step-level dims).
SCORE = {
    "Revealing_of_the_Answer": {"No": 1.0, "Yes (and the answer is correct)": 0.0, "Yes (but the answer is incorrect)": 0.0},
    "Providing_Guidance": {"Yes": 1.0, "To some extent": 0.5, "No": 0.0},
    "Actionability": {"Yes": 1.0, "To some extent": 0.5, "No": 0.0},
    "Coherence": {"Yes": 1.0, "To some extent": 0.5, "No": 0.0},
    "Tutor_Tone": {"Encouraging": 1.0, "Neutral": 0.5, "Offensive": 0.0},
    "Humanlikeness": {"Yes": 1.0, "To some extent": 0.5, "No": 0.0},
    "Step_Level_Guidance": {"Yes": 1.0, "To some extent": 0.5, "No": 0.0},
    "Load_Aware_Formatting": {"Yes": 1.0, "To some extent": 0.5, "No": 0.0},
}
DIMS = list(SCORE)

def canon(dim, v):
    m = SCORE[dim]
    if v in m:
        return m[v]
    if isinstance(v, str):
        for k in m:
            if v.strip().lower() == k.lower():
                return m[k]
        for k in m:
            if v.strip().lower().startswith(k.lower()[:6]):
                return m[k]
    return None

rows = []
for f in sorted(glob.glob("judge_out_*.json")):
    rows += json.load(open(f))

by = {s: [] for s in SETUPS}
unparsed = 0
for r in rows:
    s = key.get(r["rid"])
    if s is None:
        continue
    scored = {}
    for d in DIMS:
        sc = canon(d, r.get(d))
        if sc is None:
            unparsed += 1
        scored[d] = sc
    by[s].append(scored)

def mean(vals):
    vals = [v for v in vals if v is not None]
    return st.mean(vals) if vals else float("nan")

print(f"scored {len(rows)} candidates | unparsed label cells: {unparsed}\n")
short = {"Revealing_of_the_Answer": "NoReveal", "Providing_Guidance": "Guidance", "Actionability": "Action",
         "Coherence": "Coher", "Tutor_Tone": "Tone", "Humanlikeness": "Human",
         "Step_Level_Guidance": "StepLvl", "Load_Aware_Formatting": "LoadFmt"}
hdr = f'{"setup":<12}' + "".join(f'{short[d]:>9}' for d in DIMS) + f'{"OVERALL":>9}{"n":>4}'
print(hdr); print("-" * len(hdr))
summary = {}
for s in SETUPS:
    rs = by[s]
    dmeans = {d: mean([x[d] for x in rs]) for d in DIMS}
    ov = mean([v for x in rs for v in x.values() if v is not None])
    summary[s] = {**dmeans, "OVERALL": ov, "n": len(rs)}
    print(f'{s:<12}' + "".join(f'{dmeans[d]:>9.2f}' for d in DIMS) + f'{ov:>9.2f}{len(rs):>4}')

json.dump(summary, open("judge_summary.json", "w"), indent=2)
print("\nAll dimensions 0-1 (higher = better). OVERALL = mean of the 8 dims.")
print("wrote judge_summary.json")
