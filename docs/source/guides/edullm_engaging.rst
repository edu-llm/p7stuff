Engaging with eduLLM
====================

Pilot status
------------

The eduLLM pilot is operational. The core request-to-results flow passed
end-to-end on Issue #8, so teammates may submit validated requests; earlier
draft guidance that said to wait for Issues or pilot activation is obsolete.

Teammate onboarding
-------------------

Teammates clone or update the repository, then open it in Cursor:

.. code-block:: bash

   git clone https://github.com/edu-llm/OLMo-core.git
   cd OLMo-core
   git switch main
   git pull --ff-only origin main
   cursor .

For an existing clone, start with ``git switch main``. PR review controls
merging to ``main``. The assigned operator authorizes a job by running
``edullm run``.

Operator onboarding
-------------------

Operators clone or update the current ``main`` branch, install the W&B-enabled
CLI, authenticate GitHub and W&B, and then run:

.. code-block:: bash

   git clone https://github.com/edu-llm/OLMo-core.git
   cd OLMo-core
   git switch main
   git pull --ff-only origin main
   python -m pip install -e '.[wandb]'
   gh auth login
   wandb login
   # Connect to the MIT VPN here if your network requires it.
   edullm setup --orcd-username YOUR_MIT_USERNAME
   edullm jobs --mine

Replace ``YOUR_MIT_USERNAME`` with the operator's MIT username. Operator Slack
member IDs are centrally configured in the reviewed roster; ``edullm setup``
never asks for a Slack ID.

Meric (``meric233``) and Amy Lin (``alsy7009``) are assignment-enabled in the
reviewed roster. Each must finish or fix their local and ORCD setup before
accepting jobs; assignment eligibility does not mean that setup has succeeded.

Prepare request evidence
------------------------

The submitting teammate owns the experiment branch, including the training
code, configuration, data choices, and success metrics. The request uses a
clean non-main tree, the canonical ``edu-llm/OLMo-core`` origin, the exact full
pushed commit SHA, and script evidence. A direct-main SHA is not eligible.

PR review controls merging to ``main``. The assigned operator authorizes a job
by running ``edullm run``.

Submit with /submit-edullm-job
------------------------------

Run the real ``/submit-edullm-job`` Skill after the exact commit is pushed to
the canonical repository. The Skill validates the request, shows the canonical
Issue preview, asks for confirmation, and creates the structured Issue without
changing its validated body. Manually completing the Issue form does not
satisfy acceptance. Creating the Issue does not submit compute.

Assignment
----------

GitHub Actions validate the Issue without compute credentials, assign one
enabled operator, and send that operator one Slack assignment notification.
The workflow receives only GitHub's scoped token and the required assignment
webhook; ORCD, SSH, Kerberos, Duo, W&B, and S3 credentials stay operator-side.

Operate
-------

Operators use:

* ``edullm setup`` to configure and verify the local operator environment.
* ``edullm run`` to authorize, revalidate, and submit the oldest assigned
  eligible request.
* ``edullm jobs [--mine]`` to list authorized requests and reconcile scheduler
  state.
* ``edullm logs ISSUE`` to read the authorized bounded redacted log.
* ``edullm stop ISSUE`` to stop an authorized recorded job idempotently.
* ``edullm logout`` to close only the project SSH ControlMaster.

Identity and safety
-------------------

There are no shared credentials. Requests cannot use a direct-main SHA or
supply shell text. The exact pushed SHA is checked during validation and again
immediately before submission. Structured arguments are shell-quoted, and the
idempotent submission transaction permits exactly one ``sbatch``.

Status and tracking
-------------------

Each attempt has a deterministic W&B run ID and URL derived from its Issue,
attempt, and Slurm job ID. Training emits the real experiment metrics to that
run. ``edullm jobs`` reads ``squeue`` and ``sacct`` evidence and maps terminal
Slurm state back to the GitHub Issue.

Deferred
--------

The initial slice does not include scheduled W&B monitoring from the original
Plan 2 Task 8, Plan 3 work, S3, Apptainer, advanced Slack reminders or terminal
threads, strict ruleset automation, or broad rollout polish.
