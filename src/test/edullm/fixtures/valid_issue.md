### Purpose
Skill-DAG smoke

### Study
skill-dag-v1

### Condition
natural

### Comparison
fixed-uniform

### Commit SHA
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

### Entrypoint profile
hypothesis-smoke

### Script path
src/scripts/train/smoketests/OLMo2-190M-hypothesis-smoke.py

### Launcher
python

### Arguments JSON
["train_single", "skilldag-natural", "local", "--seed=0"]

### Data manifest
/orcd/pool/edullm/manifests/skill-dag-v1.json

### Data manifest SHA-256
bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb

### Data classification
public

### Seed
0

### W&B project
pretraining

### Success signal
20 steps and finite loss

### Success metrics
train/loss

### GPU count
1

### GPU preference
l40s

### Maximum runtime minutes
30
