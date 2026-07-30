# eduLLM CI Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore deterministic eduLLM lint checks without weakening the curated job gate.

**Architecture:** Add one repository contract test for the CI linter version, then pin Ruff to the
last known-good release in the existing development dependency list. The current CI installation
and cache-key paths consume `pyproject.toml`, so no workflow or policy change is required.

**Tech Stack:** Python, pytest, Ruff, setuptools optional dependencies, GitHub Actions.

## Global Constraints

- Pin Ruff to exactly `0.15.22`.
- Do not change `config/edullm/policy.yaml` or its eleven required checks.
- Do not add external credentials to GitHub Actions.
- Do not attempt the separate Ruff 0.16.0 migration.
- Keep the repair limited to the dependency contract, its focused test, and approved design/plan
  documentation.

---

### Task 1: Pin the CI Linter

**Files:**
- Modify: `pyproject.toml:39-63`
- Modify: `src/test/edullm/task_4_config_test.py:12-20`
- Test: `src/test/edullm/task_4_config_test.py`

**Interfaces:**
- Consumes: `[project.optional-dependencies].dev`, installed by
  `.github/actions/setup-python-env/action.yml` through `uv pip install -e .[all]`.
- Produces: the exact dependency string `ruff==0.15.22` for local and GitHub Actions environments.

- [ ] **Step 1: Write the failing dependency-contract test**

Add the project metadata path beside the existing configuration path constants:

```python
PYPROJECT = Path("pyproject.toml")
```

Add this focused test:

```python
def test_ci_linter_dependency_is_exactly_pinned():
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    dev_dependencies = pyproject.partition("[project.optional-dependencies]\n")[2].partition(
        "\nbeaker ="
    )[0]

    assert '"ruff==0.15.22",' in dev_dependencies
    assert '\n    "ruff",' not in dev_dependencies
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest -q src/test/edullm/task_4_config_test.py::test_ci_linter_dependency_is_exactly_pinned
```

Expected: FAIL because `pyproject.toml` currently contains the unbounded dependency `"ruff"`.

- [ ] **Step 3: Pin Ruff in the development dependency list**

Change:

```toml
    "ruff",
```

to:

```toml
    "ruff==0.15.22",
```

- [ ] **Step 4: Install and verify the pinned tool**

Run:

```bash
python -m pip install "ruff==0.15.22"
ruff --version
```

Expected final line:

```text
ruff 0.15.22
```

- [ ] **Step 5: Run focused and regression verification**

Run:

```bash
pytest -q src/test/edullm/task_4_config_test.py::test_ci_linter_dependency_is_exactly_pinned
pytest -q src/test/edullm
make lint-check
make style-check
```

Expected:

- focused dependency test passes;
- all 995 eduLLM tests pass;
- Ruff reports `All checks passed!`;
- isort and Black report no changes required.

- [ ] **Step 6: Commit the repair**

```bash
git add pyproject.toml src/test/edullm/task_4_config_test.py
git commit -m "fix: pin edullm ci linter"
```

- [ ] **Step 7: Publish and verify GitHub CI**

```bash
git push -u origin edullm-ci-stability
gh pr create \
  --repo edu-llm/OLMo-core \
  --base main \
  --head edullm-ci-stability \
  --title "Stabilize eduLLM CI lint gate" \
  --body "Pins Ruff to the last known-good CI version while retaining all curated eduLLM checks."
```

Verify that `Lint` and `Lint (min Python)` both install Ruff 0.15.22 and pass. Do not merge until a
different listed approver approves the exact pull-request head and all eleven curated checks pass.
