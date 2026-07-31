"""Super-NaturalInstructions prompt pool: source, filters, round-robin sampling (PLAN §3).

Source (verified 2026-07-31, recorded in every manifest)
--------------------------------------------------------
Authoritative: GitHub ``allenai/natural-instructions`` at the pinned commit below —
``tasks/<name>.json`` (1,616 tasks) + ``splits/default/{train,test}_tasks.txt``
(756 train / 119 test task ids).

The HF mirror ``Muennighoff/natural-instructions`` **does exist**, but its
preprocessed records carry only ``{task_name, id, definition, inputs, targets}``.
It drops the ``Source`` and ``Categories`` fields, which PLAN §3 filter 2 needs for
the mandatory contamination exclusion and which §3 filter 3 needs for the category
histogram. So it is *not* usable as the primary source; ``--source hf`` exists only
as an escape hatch and refuses to run the Source filter silently.

Fetching
--------
Task JSONs are large (task001 alone is 11 MB) and we need at most a few hundred
instances per task, so the default backend streams each file and stops once it has
the metadata header plus ``--instances_per_task`` instances. That turns a ~2 GB
pull into ~200 MB. ``--superni_dir`` (a local ``git clone --depth 1``) uses the same
reader against the filesystem and is the recommended path on the cluster.

Because the stream stops early, the instances we see are the file's *first* K, not
a random sample. Within a SuperNI task all instances share one template, so the
mean-gold-length estimate is stable; ``--instances_per_task 0`` reads every instance
if you want to remove the assumption.
"""

from __future__ import annotations

import codecs
import json
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from .textutil import word_count

# Pinned commit — the same one the HF mirror's own get_ni.py pulls from.
SUPERNI_COMMIT = "6174af63465999768fbc09f5dd8a7f1a5dfe9abc"
SUPERNI_REPO = "allenai/natural-instructions"
RAW_BASE = f"https://raw.githubusercontent.com/{SUPERNI_REPO}/{SUPERNI_COMMIT}"

# PLAN §3 filter 2. Deliberately broad: `math_eval/` grades BBH-logical-deduction
# and BBH ⊂ BIG-Bench, and over-dropping a few legitimate tasks is cheap while
# contaminating an eval set is fatal. Matched against the *normalised concatenation*
# of task name + Source + URL + Categories + Domains.
CONTAM_PATTERNS = (
    "big-bench", "big_bench", "bigbench", "big bench", "bbh",
    "gsm8k", "gsm-8k", "gsm_8k", "grade school math",
    "math",          # catches mathqa, math_dataset, hendrycks math, mathematics, ...
    "aime",
)

MIN_GOLD_WORDS = 30      # PLAN §3 filter 3

_NONWORD = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# Streaming reader: metadata header + first K instances, without buffering 11 MB.
# ---------------------------------------------------------------------------
class _StrStream:
    """Incremental UTF-8 view over a binary stream, with a trimmable buffer."""

    def __init__(self, raw, chunk: int = 1 << 16):
        self._raw = raw
        self._dec = codecs.getincrementaldecoder("utf-8")()
        self._chunk = chunk
        self.buf = ""
        self.eof = False

    def fill(self) -> bool:
        if self.eof:
            return False
        data = self._raw.read(self._chunk)
        if not data:
            self.eof = True
            self.buf += self._dec.decode(b"", final=True)
            return False
        self.buf += self._dec.decode(data)
        return True

    def trim(self, i: int) -> None:
        if i:
            self.buf = self.buf[i:]


def _read_task_stream(raw, max_instances: int) -> tuple[dict, list[dict], bool]:
    """Parse a SuperNI task file. Returns ``(meta, instances, complete)``.

    ``complete`` is False when we stopped early at ``max_instances``.
    """
    s = _StrStream(raw)
    key = '"Instances"'

    # 1. Metadata: everything before the "Instances" key. SuperNI writes that key
    #    last, so the prefix closes into a valid object.
    while key not in s.buf:
        if not s.fill():
            break
        if len(s.buf) > (16 << 20):
            raise ValueError("no 'Instances' key in the first 16 MB — unexpected task format")
    idx = s.buf.find(key)
    if idx < 0:
        doc = json.loads(s.buf)
        inst = doc.get("Instances", [])
        return doc, (inst[:max_instances] if max_instances else inst), True

    head = s.buf[:idx].rstrip().rstrip(",")
    try:
        meta = json.loads(head + "}")
    except json.JSONDecodeError:  # fall back to a full parse rather than guess
        while s.fill():
            pass
        doc = json.loads(s.buf)
        inst = doc.get("Instances", [])
        return doc, (inst[:max_instances] if max_instances else inst), True

    # 2. Instances: incremental object-by-object decode of the array.
    lb = s.buf.find("[", idx + len(key))
    if lb < 0:
        return meta, [], True
    s.trim(lb + 1)

    dec = json.JSONDecoder()
    instances: list[dict] = []
    complete = False
    while True:
        if max_instances and len(instances) >= max_instances:
            break
        i = 0
        while True:
            while i < len(s.buf) and s.buf[i] in " \t\r\n,":
                i += 1
            if i < len(s.buf):
                break
            if not s.fill():
                complete = True
                break
        if complete or i >= len(s.buf):
            complete = True
            break
        if s.buf[i] == "]":
            complete = True
            break
        while True:
            try:
                obj, end = dec.raw_decode(s.buf, i)
                break
            except json.JSONDecodeError:
                if not s.fill():
                    raise
        instances.append(obj)
        s.trim(end)
    return meta, instances, complete


def _http_get(url: str, retries: int = 4, timeout: int = 120):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "impl4-ssd/1.0"})
            return urllib.request.urlopen(req, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as e:  # pragma: no cover - network
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} attempts: {url}") from last


# ---------------------------------------------------------------------------
# Source abstraction
# ---------------------------------------------------------------------------
@dataclass
class SuperNISource:
    """Where task files come from. ``local_dir`` wins when set.

    ``cache_dir`` stores each task's metadata + the instances we actually read, so
    re-scanning at a different ``--min_gold_words`` costs nothing. The cache key
    includes ``instances_per_task``, so raising it correctly forces a refetch.
    """

    local_dir: Optional[Path] = None
    commit: str = SUPERNI_COMMIT
    instances_per_task: int = 300
    cache_dir: Optional[Path] = None

    def describe(self) -> dict:
        """Provenance block for the manifest — "which source did you actually use"."""
        return {
            "repo": SUPERNI_REPO,
            "commit": self.commit,
            "mode": "local_clone" if self.local_dir else "github_raw_stream",
            "local_dir": str(self.local_dir) if self.local_dir else None,
            "instances_per_task": self.instances_per_task,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
        }

    def split_task_names(self, split: str) -> list[str]:
        assert split in ("train", "test")
        if self.local_dir:
            p = self.local_dir / "splits" / "default" / f"{split}_tasks.txt"
            text = p.read_text(encoding="utf-8")
        else:
            with _http_get(f"{RAW_BASE}/splits/default/{split}_tasks.txt") as r:
                text = r.read().decode("utf-8")
        return [ln.strip() for ln in text.splitlines() if ln.strip()]

    def _cache_path(self, name: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return Path(self.cache_dir) / f"{name}.k{self.instances_per_task}.json.gz"

    def read_task(self, name: str) -> tuple[dict, list[dict], bool]:
        cache = self._cache_path(name)
        if cache is not None and cache.exists():
            import gzip
            with gzip.open(cache, "rt", encoding="utf-8") as f:
                d = json.load(f)
            return d["meta"], d["instances"], d["complete"]

        if self.local_dir:
            with open(self.local_dir / "tasks" / f"{name}.json", "rb") as f:
                out = _read_task_stream(f, self.instances_per_task)
        else:
            with _http_get(f"{RAW_BASE}/tasks/{name}.json") as r:
                out = _read_task_stream(r, self.instances_per_task)

        if cache is not None:
            import gzip
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                json.dump({"meta": out[0], "instances": out[1], "complete": out[2]}, f)
            tmp.replace(cache)
        return out


# ---------------------------------------------------------------------------
# Filters (PLAN §3, applied in order)
# ---------------------------------------------------------------------------
@dataclass
class TaskInfo:
    name: str
    definition: str
    source: list[str]
    categories: list[str]
    domains: list[str]
    n_instances_seen: int
    mean_gold_words: float
    instances: list[dict] = field(default_factory=list)


def _norm_field(*values) -> str:
    parts: list[str] = []
    for v in values:
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, (list, tuple)):
            parts.extend(str(x) for x in v)
    return _NONWORD.sub(" ", " ".join(parts).lower()).strip()


def contamination_hit(meta: dict, task_name: str,
                      patterns: tuple[str, ...] = CONTAM_PATTERNS) -> Optional[str]:
    """PLAN §3 filter 2a — task-level source/name exclusion."""
    hay = " " + _norm_field(
        task_name, meta.get("Source"), meta.get("URL"),
        meta.get("Categories"), meta.get("Domains"),
    ) + " "
    for pat in patterns:
        norm_pat = _NONWORD.sub(" ", pat.lower()).strip()
        if not norm_pat:
            continue
        if f" {norm_pat} " in hay:
            return pat
    return None


def is_english_task(meta: dict) -> bool:
    """PLAN §3 filter 1 — English training tasks only."""
    def ok(key: str) -> bool:
        vals = meta.get(key) or []
        if isinstance(vals, str):
            vals = [vals]
        return bool(vals) and all("english" in str(v).lower() for v in vals)

    return ok("Input_language") and ok("Output_language") and ok("Instruction_language")


def gold_of(instance: dict) -> str:
    out = instance.get("output")
    if isinstance(out, list):
        return str(out[0]) if out else ""
    return str(out or "")


def mean_gold_words(instances: list[dict]) -> float:
    if not instances:
        return 0.0
    return sum(word_count(gold_of(i)) for i in instances) / len(instances)


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------
def scan_tasks(
    source: SuperNISource,
    split: str = "train",
    ngram_index=None,
    min_gold_words: int = MIN_GOLD_WORDS,
    contam_patterns: tuple[str, ...] = CONTAM_PATTERNS,
    max_tasks: int = 0,
    log=print,
) -> tuple[list[TaskInfo], dict]:
    """Fetch + filter tasks. Returns ``(retained_tasks, stats)``.

    ``ngram_index`` is applied per *instance* (PLAN §3 filter 2b); an instance whose
    definition/input/gold shares a 13-gram with an eval prompt is dropped, and a task
    left with no instances is dropped with it.
    """
    names = source.split_task_names(split)
    if max_tasks:
        names = names[:max_tasks]

    stats = {
        "split": split,
        "tasks_listed": len(names),
        "dropped_fetch_error": 0,
        "dropped_non_english": 0,
        "dropped_contaminated_source": 0,
        "dropped_short_gold": 0,
        "dropped_no_instances": 0,
        "instances_dropped_ngram": 0,
        "contaminated_tasks": [],
        "fetch_errors": [],
    }
    retained: list[TaskInfo] = []
    # Mean gold length of every task that clears filters 1-2, so the length threshold
    # can be read off a measured distribution instead of guessed. SuperNI is dominated
    # by short-answer classification, and PLAN §3 filter 3 bites much harder than the
    # plan's prose implies — this is what lets you see by how much.
    length_profile: list[tuple[str, float, int]] = []

    for i, name in enumerate(names, 1):
        try:
            meta, instances, _complete = source.read_task(name)
        except Exception as e:  # pragma: no cover - network/IO
            stats["dropped_fetch_error"] += 1
            stats["fetch_errors"].append(f"{name}: {type(e).__name__}: {e}")
            continue

        if not is_english_task(meta):
            stats["dropped_non_english"] += 1
            continue

        pat = contamination_hit(meta, name, contam_patterns)
        if pat:
            stats["dropped_contaminated_source"] += 1
            stats["contaminated_tasks"].append({"task": name, "pattern": pat,
                                                "source": meta.get("Source")})
            continue

        definition = (meta.get("Definition") or [""])[0]

        if ngram_index is not None:
            clean = []
            for inst in instances:
                probe = f"{definition}\n{inst.get('input', '')}\n{gold_of(inst)}"
                if ngram_index.hit(probe) is None:
                    clean.append(inst)
                else:
                    stats["instances_dropped_ngram"] += 1
            instances = clean

        if not instances:
            stats["dropped_no_instances"] += 1
            continue

        mgw = mean_gold_words(instances)
        length_profile.append((name, round(mgw, 2), len(instances)))
        if mgw < min_gold_words:
            stats["dropped_short_gold"] += 1
            continue

        retained.append(TaskInfo(
            name=name,
            definition=definition,
            source=list(meta.get("Source") or []),
            categories=list(meta.get("Categories") or []),
            domains=list(meta.get("Domains") or []),
            n_instances_seen=len(instances),
            mean_gold_words=mgw,
            instances=instances,
        ))
        if i % 50 == 0:
            log(f"  scanned {i}/{len(names)} tasks | retained {len(retained)}")

    stats["tasks_retained"] = len(retained)
    stats["category_histogram"] = dict(Counter(
        c for t in retained for c in (t.categories or ["(none)"])
    ).most_common())
    stats["instances_available"] = sum(t.n_instances_seen for t in retained)
    stats["min_gold_words"] = min_gold_words
    stats["length_profile"] = length_profile_summary(length_profile)
    return retained, stats


LENGTH_THRESHOLDS = (5, 10, 15, 20, 25, 30, 40, 50, 75, 100)


def length_profile_summary(profile: list[tuple[str, float, int]]) -> dict:
    """Retention at a range of ``min_gold_words`` thresholds, plus the raw quantiles.

    Reported so the ≥30-word choice in PLAN §3 is a *measured* trade-off between
    "the gold arm carries real training weight" and "domain breadth is preserved",
    rather than a number that silently collapses the pool to a handful of tasks.
    """
    if not profile:
        return {"n_tasks_scored": 0}
    vals = sorted(p[1] for p in profile)

    def q(f: float) -> float:
        return vals[min(len(vals) - 1, int(f * (len(vals) - 1)))]

    return {
        "n_tasks_scored": len(vals),
        "quantiles": {"p10": q(0.10), "p25": q(0.25), "p50": q(0.50),
                      "p75": q(0.75), "p90": q(0.90), "max": vals[-1]},
        "tasks_retained_at_threshold": {
            str(t): sum(1 for v in vals if v >= t) for t in LENGTH_THRESHOLDS
        },
        "instances_available_at_threshold": {
            str(t): sum(n for _, v, n in profile if v >= t) for t in LENGTH_THRESHOLDS
        },
        "longest_tasks": sorted(profile, key=lambda p: -p[1])[:15],
    }


def round_robin_sample(tasks: list[TaskInfo], n: int, seed: int) -> list[dict]:
    """PLAN §3 filter 4 — one instance at a time across tasks, so none dominates."""
    rng = random.Random(seed)
    order = sorted(tasks, key=lambda t: t.name)
    pools: list[tuple[TaskInfo, list[dict]]] = []
    for t in order:
        inst = list(t.instances)
        rng.shuffle(inst)
        pools.append((t, inst))
    rng.shuffle(pools)

    out: list[dict] = []
    cursor = 0
    while len(out) < n:
        progressed = False
        for t, inst in pools:
            if cursor < len(inst):
                item = inst[cursor]
                out.append({
                    "superni_task_id": t.name,
                    "instance_id": item.get("id"),
                    "definition": t.definition,
                    "input": item.get("input", ""),
                    "gold": gold_of(item),
                    "categories": t.categories,
                    "source_datasets": t.source,
                })
                progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            break
        cursor += 1
    return out


def user_message(item: dict) -> str:
    """PLAN §4: ``task_definition + "\\n\\n" + instance_input``, in the *user* turn.

    Never the system slot — the Impl 2 contract is "system message present ⇔ tutor
    mode", and a non-pedagogy system message would redefine that switch.
    """
    return f"{item['definition']}\n\n{item['input']}"


def iter_pool(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
