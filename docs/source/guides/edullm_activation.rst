eduLLM activation administration
================================

The eduLLM pilot is operational. Current teammate and operator onboarding lives
in :doc:`edullm_engaging`; earlier instructions to wait for Issue support or
pilot activation are obsolete. PR review controls merging to ``main``. The
assigned operator authorizes a job by running ``edullm run``. The sections
below retain the administrator-only activation boundary for reference. Never
put the Slack webhook or any compute or experiment credential in Git or chat.

Agent can prepare
-----------------

After the user supplies the three public identities, an agent can prepare:

* a diff containing one public team-lead GitHub login, one public operator
  GitHub login and Slack member ID, the three core workflow guard changes,
  focused tests, and documentation;
* static config, workflow, and credential-boundary evidence; and
* the exact label and secret commands below for the user to inspect.

The three public inputs are:

* ``TEAM_LEAD_GITHUB_LOGIN``
* ``OPERATOR_GITHUB_LOGIN``
* ``OPERATOR_SLACK_USER_ID``

Do not supply the Slack webhook, ORCD, SSH, Kerberos, Duo, W&B, or S3
credentials to the agent.

User must perform
-----------------

The user must:

* choose and provide the three public identity values;
* configure ``SLACK_WEBHOOK_URL`` directly in GitHub, never in Git or chat;
* create or confirm the required labels with repository administration
  permission;
* review and merge the activation change; and
* provide team-lead merge review plus operator GitHub, SSH, Kerberos, Duo, and
  W&B access at the live stop points.

User-only repository commands
-----------------------------

These commands are for a user with repository permission to inspect and run.
They are documentation, not an activation script:

.. code-block:: bash

   gh label create edullm-job --repo edu-llm/OLMo-core --color 1D76DB --force
   gh label create status:requested --repo edu-llm/OLMo-core --color D4C5F9 --force
   gh label create status:validating --repo edu-llm/OLMo-core --color D4C5F9 --force
   gh label create status:ready --repo edu-llm/OLMo-core --color 0E8A16 --force
   gh label create status:assigned --repo edu-llm/OLMo-core --color 0E8A16 --force
   gh label create status:submitted --repo edu-llm/OLMo-core --color FBCA04 --force
   gh label create status:running --repo edu-llm/OLMo-core --color FBCA04 --force
   gh label create status:completed --repo edu-llm/OLMo-core --color 0E8A16 --force
   gh label create status:failed --repo edu-llm/OLMo-core --color D93F0B --force
   gh label create status:cancelled --repo edu-llm/OLMo-core --color 6A737D --force
   gh label create status:preempted --repo edu-llm/OLMo-core --color D93F0B --force

   read -r -s SLACK_WEBHOOK_URL
   printf %s "$SLACK_WEBHOOK_URL" | \
     gh secret set SLACK_WEBHOOK_URL --repo edu-llm/OLMo-core
   unset SLACK_WEBHOOK_URL

The user can confirm only the secret name and label metadata without exposing
the webhook value:

.. code-block:: bash

   gh secret list --repo edu-llm/OLMo-core | \
     awk '$1 == "SLACK_WEBHOOK_URL" {found=1} END {exit !found}'
   gh label list --repo edu-llm/OLMo-core --limit 100 \
     --json name,color,description

Activation boundary
-------------------

The activation diff may remove only the validation, reusable assignment
handoff, and assignment workflow literal disables. Reminders, reassignment,
terminal Slack, scheduled W&B monitoring, all Plan 3 work, and strict ruleset
automation remain disabled or deferred. GitHub Actions receive only
``github.token`` and the assignment Slack webhook; all compute and experiment
credentials remain operator-side.
