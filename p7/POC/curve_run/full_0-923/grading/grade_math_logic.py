"""Grade math+logic results with the frontier rubric: verifiable final-answer accuracy.

Stage 1 (this script): deterministic checking.
  - int (GSM8K/AIME): parse final integer, exact match.
  - mc  (BBH)       : parse choice letter (A-G), exact match.
  - expr (MATH-500) : normalize LaTeX and string-compare; sympy equivalence if installed.
Items of type 'expr' that are NOT a clean string/sympy match are written to needs_verify.json
for Stage 2 (subagent LLM-as-verifier), so we never under-count equivalent-but-differently-written
answers. Run build_verify_batches.py -> spawn judges -> rerun this script with --with-verify.

Usage:
  python grade_math_logic.py [results.jsonl] [--with-verify]
Output/verify filenames are tagged from the results filename so multiple runs (e.g. no-SI vs
direct-SI) don't clobber each other:
  math_logic_results.jsonl            -> tag 'nosi'
  math_logic_results_directsi.jsonl   -> tag 'directsi'
"""
import json, re, sys, glob
from collections import defaultdict

WITH_VERIFY = "--with-verify" in sys.argv
_pos = [a for a in sys.argv[1:] if not a.startswith("--")]
RES_PATH = _pos[0] if _pos else "math_logic_results.jsonl"
_m = re.search(r"math_logic_results_(\w+?)(?:_v\d+.*)?\.jsonl", RES_PATH)
TAG = _m.group(1) if _m else "nosi"
GRADED_OUT = f"math_logic_graded_{TAG}.json"
NEEDS_OUT = f"needs_verify_{TAG}.json"
VERIFIER_GLOB = f"verifier_out_{TAG}_*.json"

PROMPTS = {r["id"]: r for r in (json.loads(l) for l in open("math_logic_prompts.jsonl"))}
RES = [json.loads(l) for l in open(RES_PATH)]
print(f"grading {RES_PATH}  (tag={TAG})")

try:
    import sympy
    from sympy.parsing.latex import parse_latex
    HAVE_SYMPY = True
except Exception:
    HAVE_SYMPY = False


def last_boxed(s):
    i = s.rfind(r"\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
    return None


def extract(resp, atype):
    resp = resp or ""
    if atype == "mc":
        box = last_boxed(resp)
        if box:
            lm = re.findall(r"[A-G]", box.upper())
            if lm:
                return lm[-1]
        m = re.findall(r"answer is\s*\(?\s*([A-G])\s*\)?", resp, re.I) or \
            re.findall(r"final answer\s*[:\-]?\s*\(?\s*([A-G])\s*\)?", resp, re.I) or \
            re.findall(r"\(([A-G])\)", resp) or re.findall(r"\b([A-G])\b", resp[-40:])
        return m[-1].upper() if m else None
    box = last_boxed(resp)
    cand = box
    if cand is None:
        m = re.findall(r"(?:final answer|answer)\s*(?:is|:)?\s*\$?([^\n$]+)", resp, re.I)
        cand = m[-1].strip() if m else None
    if atype == "int":
        src = cand if cand is not None else resp
        nums = re.findall(r"-?\d[\d,]*", src)
        return nums[-1].replace(",", "") if nums else None
    return cand  # expr


def norm_expr(x):
    if x is None:
        return None
    x = x.strip().strip("$").strip()
    x = x.replace(r"\left", "").replace(r"\right", "").replace(r"\!", "").replace(r"\,", "")
    x = x.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    x = re.sub(r"\\text\{[^}]*\}", "", x)
    x = x.replace(" ", "").replace(r"\%", "").rstrip(".")
    x = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", x)
    return x


def sympy_eq(a, b):
    if not HAVE_SYMPY:
        return None
    try:
        return bool(sympy.simplify(parse_latex(a) - parse_latex(b)) == 0)
    except Exception:
        try:
            return sympy.nsimplify(a) == sympy.nsimplify(b)
        except Exception:
            return None


def check(atype, gold, cand):
    """Return True/False, or None if 'unsure -> send to verifier' (expr only)."""
    if cand is None:
        return None if atype == "expr" else False
    if atype == "int":
        try:
            return int(float(cand)) == int(float(gold))
        except Exception:
            return False
    if atype == "mc":
        g = re.sub(r"[()]", "", gold).strip().upper()
        return cand.strip().upper() == g
    # expr
    if norm_expr(cand) == norm_expr(gold):
        return True
    se = sympy_eq(cand, gold)
    if se is True:
        return True
    return None  # let a verifier decide


verifier = {}
if WITH_VERIFY:
    for f in glob.glob(VERIFIER_GLOB):
        for r in json.load(open(f)):
            verifier[r["task_id"]] = str(r.get("verdict", "")).lower().startswith(("correct", "true", "yes"))

per_model = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # model -> source -> [correct, n]
needs_verify = []
detail = []
for r in RES:
    meta = PROMPTS[r["id"]]
    gold, atype, src = meta["gold"], meta["answer_type"], meta["source"]
    for model in ("base", "sft"):
        resp = r["outputs"].get(model, "")
        cand = extract(resp, atype)
        res = check(atype, gold, cand)
        if res is None:
            tid = f"{r['id']}__{model}"
            if WITH_VERIFY and tid in verifier:
                res = verifier[tid]
            else:
                needs_verify.append({"task_id": tid, "id": r["id"], "model": model,
                                     "question": meta["prompt"], "gold": gold,
                                     "candidate": resp})
                res = False  # provisional until verified
        per_model[model][src][1] += 1
        per_model[model][src][0] += int(bool(res))
        detail.append({"id": r["id"], "model": model, "source": src, "gold": gold,
                       "extracted": cand, "correct": bool(res)})

sources = ["GSM8K", "MATH-500", "BBH-logical_deduction", "AIME-2024"]
print("=" * 66)
print("MATH + LOGIC ACCURACY (final-answer, frontier rubric)   base vs SFT")
print("=" * 66)
print(f"{'source':<26}{'base':>10}{'sft':>10}")
for s in sources:
    b, n = per_model["base"][s]
    sc, _ = per_model["sft"][s]
    print(f"{s:<26}{f'{b}/{n} ({100*b/n:.0f}%)':>10}{f'{sc}/{n} ({100*sc/n:.0f}%)':>10}")
tb = sum(per_model["base"][s][0] for s in sources); tn = sum(per_model["base"][s][1] for s in sources)
ts = sum(per_model["sft"][s][0] for s in sources)
print("-" * 46)
print(f"{'OVERALL':<26}{f'{tb}/{tn} ({100*tb/tn:.0f}%)':>10}{f'{ts}/{tn} ({100*ts/tn:.0f}%)':>10}")

json.dump(detail, open(GRADED_OUT, "w"), indent=2, ensure_ascii=False)
if needs_verify and not WITH_VERIFY:
    json.dump(needs_verify, open(NEEDS_OUT, "w"), indent=2, ensure_ascii=False)
    print(f"\n{len(needs_verify)} MATH-500 answers need subagent verification (symbolic equivalence).")
    print(f"-> python build_verify_batches.py {NEEDS_OUT}, spawn verifiers (verifier_out_{TAG}_*.json),")
    print(f"   then: python grade_math_logic.py {RES_PATH} --with-verify")
else:
    print(f"\ngrading complete. wrote {GRADED_OUT}")
