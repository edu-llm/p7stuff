# SocraTeach SFT on MIT ORCD (Engaging / SLURM)

Cluster port of `olmo2_1b_sft_colab.ipynb`. Same recipe (LearnLM-style per-dialogue
System Instructions + co-training, LoRA, assistant-only loss masking), as `train.py` +
`sbatch` instead of a notebook. The **only** substantive change vs. the base-model run is
`--start_from instruct` (`allenai/OLMo-2-0425-1B-Instruct`), with outputs tagged `-instruct`.

This folder is **self-contained** — the prepared data is included under `data/`, so you can
copy just this one folder to the cluster.

| File | Purpose |
|---|---|
| `setup_orcd_env.sh` | one-time conda env + deps (run on a login node) |
| `train_sft.py` | data load → LoRA → masking → `Trainer` → save |
| `run_sft.sbatch` | submits the training job on `mit_normal_gpu` (1× L40S) |
| `generate_test_results.py` | 2×2 factorial outputs — **for later; evals are out of scope now** |
| `data/socrateach_sft_{train,val,test}.jsonl` | prepared splits (train = 30k mix). Do not regenerate. |

---

## Step 0 — Set up SSH access (one time, on your Mac)

You already have an SSH key at `~/.ssh/id_ed25519`. Get it onto Engaging so logins/transfers
are smoother (you'll still do Duo the first time). Replace `<KERBEROS>` with your MIT
Kerberos username:

```bash
# Copies your public key to Engaging's authorized_keys. Enter Kerberos password + approve Duo.
ssh-copy-id <KERBEROS>@orcd-login.mit.edu

# Test it:
ssh <KERBEROS>@orcd-login.mit.edu
```

**Fewer Duo prompts (recommended):** add this to `~/.ssh/config` on your Mac so repeated
connections reuse one authenticated channel:

```
Host orcd
    HostName orcd-login.mit.edu
    User <KERBEROS>
    IdentityFile ~/.ssh/id_ed25519
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
```

Then you can just `ssh orcd` and `rsync ... orcd:...`. (Note: ORCD still requires Kerberos +
Duo on the *first* connection of a session; the key + control channel reduce how often.)

## Step 1 — Copy this folder to Engaging

From your **Mac** (not a login node):

```bash
rsync -avP ~/Documents/MericXing/MIT/Intern/AlphaAI/P7_Inference_Engineering/ORCD-SFT orcd:~/
```

## Step 2 — One-time environment setup (login node)

```bash
ssh orcd
cd ~/ORCD-SFT
bash setup_orcd_env.sh          # installs Miniforge + conda env "socrateach"
```

This also uninstalls `torchao` (an old version breaks `peft.get_peft_model`).

## Step 3 — (Recommended) smoke test on a GPU

```bash
salloc -p mit_normal_gpu -G l40s:1 -c 16 --mem=64G -t 01:00:00
source ~/miniforge3/etc/profile.d/conda.sh && conda activate socrateach
cd ~/ORCD-SFT
python train_sft.py --start_from instruct --poc      # tiny run, ~minutes
exit
```

## Step 4 — Full training run (batch)

```bash
cd ~/ORCD-SFT
# optional: cache HF downloads on big shared storage instead of home
export HF_HOME=/orcd/pool/<yourpath>/hf_cache
sbatch run_sft.sbatch
```

Monitor:

```bash
squeue -u $USER
tail -f logs/olmo2_sft_instruct_<jobid>.out
```

Produces `olmo2-1b-socratic-tutor-instruct/` — the LoRA adapter + tokenizer.

---

## GPU / memory sizing (ORCD free tier)

- **Partition:** `mit_normal_gpu`, **1× L40S** (44 GB). A 1B LoRA needs only one GPU; the free
  base tier allows up to **2 GPUs / 6h**, so this is well within limits and needs no
  multi-GPU/distributed setup.
- Gradient checkpointing **ON** (`use_reentrant=False`) + `use_cache=False`,
  `per_device_batch=8`, `grad_accum=4` (effective 32). Fits ~44 GB comfortably.
- If you ever OOM: `--per_device_batch 4 --grad_accum 8` (same effective batch), or lower
  `--max_len`.
- One epoch over 30k examples runs in roughly ~1h on an L40S, so a single 4h job is enough;
  `--resume auto` is wired in as a safety net for the 6h wall-time.

## Tunable knobs (`train_sft.py`)

`--num_epochs` (1), `--learning_rate` (2e-4 LoRA), `--per_device_batch`, `--grad_accum`,
`--max_len` (1024), `--train_total` (0 = whole file), `--lora_r/alpha/dropout`,
`--full_finetune`, `--poc`, `--resume auto`.
