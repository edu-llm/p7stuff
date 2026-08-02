"""Put ``../impl4_ssd`` on ``sys.path`` and re-export the machinery Impl 5 shares with it.

**Import rather than copy, and the reason is not convenience** (PLAN §10 "Reuse policy"):
both implementations' arms go onto the same KL–forgetting plane and are graded by the same
driver. A silent divergence in tokenisation, token-matching or the degeneracy rules between
Impl 4 and Impl 5 would invalidate that comparison *without producing an error anywhere*.

``impl4/paths.py`` resolves ``IMPL4_ROOT`` from its own file, so importing it from here keeps
``POC_ROOT`` pointing at the real POC tree rather than at ``impl5_ssd/``.

The risk of the coupling is that Impl 5 breaks when Impl 4 changes. ``tests/test_impl5.py``
pins the behaviours depended on here, and ``acceptance_checks5.py`` fails loudly.
"""

from __future__ import annotations

import sys
from pathlib import Path

IMPL5_ROOT = Path(__file__).resolve().parents[1]
POC_ROOT = IMPL5_ROOT.parent
IMPL4_ROOT = POC_ROOT / "impl4_ssd"

if not (IMPL4_ROOT / "impl4" / "__init__.py").exists():   # pragma: no cover
    raise ImportError(
        f"Impl 5 reuses Impl 4's library but {IMPL4_ROOT}/impl4 is not there. "
        f"Impl 5 is not standalone by design — see impl5/_impl4.py."
    )
if str(IMPL4_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPL4_ROOT))

from impl4 import (                                       # noqa: E402
    chat,
    degeneracy,
    gate,
    generate,
    manifest,
    mixing,
    ngram,
    paths,
    tulu,
)
from impl4 import config as config4                       # noqa: E402

__all__ = [
    "IMPL4_ROOT", "IMPL5_ROOT", "POC_ROOT",
    "chat", "config4", "degeneracy", "gate", "generate", "manifest", "mixing",
    "ngram", "paths", "tulu",
]
