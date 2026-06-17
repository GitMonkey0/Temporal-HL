# Joint Student Notes

This file is an experiment log, not paper text.

## Goal

Test whether temporal HL can help a raw 3D-joint student beyond symbolic-only
experiments.

The intended comparison is:

- raw-joint scratch baseline
- raw-joint pretrain on train labels only
- raw-joint student with temporal-HL teacher supervision

## Implementation status

Script:

- [train_joint_student.py](/opt/tiger/hand/tools/train_joint_student.py)

Key fixes completed:

- teacher features are now merged from temporal HL channels before use
- teacher vectors are attached to joint windows by `sample_id`
- the script raises an explicit error if any joint window is missing its teacher
- short joint windows are padded to fixed length by repeating the last frame
- pretrain and finetune teacher weights are now controlled separately

## Data paths used

Joint supervision source:

- `/opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_train_joint_3d.json`
- `/opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_val_joint_3d.json`
- `/opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_test_joint_3d.json`

Symbolic source:

- [temporal_hl_train.json](/opt/tiger/hand/experiments/generated/temporal_hl_train.json)
- [temporal_hl_val.json](/opt/tiger/hand/experiments/generated/temporal_hl_val.json)
- [temporal_hl_test.json](/opt/tiger/hand/experiments/generated/temporal_hl_test.json)

## Smoke run

Command:

```bash
python tools/train_joint_student.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --train-pretrain-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_train_joint_3d.json \
  --train-finetune-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_val_joint_3d.json \
  --test-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_test_joint_3d.json \
  --fractions 1.0 \
  --seeds 0 \
  --pretrain-epochs 3 \
  --finetune-epochs 3 \
  --hidden-dim 64 \
  --teacher-weight 0.5 \
  --output experiments/generated/joint_student_smoke.json
```

Result:

- [joint_student_smoke.json](/opt/tiger/hand/experiments/generated/joint_student_smoke.json)
- baseline sequence accuracy: `0.2308`
- teacher-guided sequence accuracy: `0.2308`
- baseline window accuracy: `0.2804`
- teacher-guided window accuracy: `0.3176`

Interpretation:

- the pipeline runs end to end
- teacher supervision changes learning behavior
- but this tiny run is too weak to support any positive claim

## Main controlled runs

Shared setup:

- window size `32`
- stride `16`
- hidden dim `128`
- pretrain epochs `20`
- finetune epochs `20`
- fraction `1.0`
- seeds `0, 1, 2`

### 1. Scratch baseline vs teacher-guided pretrain+finetune

Command:

```bash
python tools/train_joint_student.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --train-pretrain-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_train_joint_3d.json \
  --train-finetune-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_val_joint_3d.json \
  --test-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_test_joint_3d.json \
  --fractions 1.0 \
  --seeds 0 1 2 \
  --pretrain-epochs 20 \
  --finetune-epochs 20 \
  --hidden-dim 128 \
  --teacher-weight 0.5 \
  --output experiments/generated/joint_student_temporal_v1.json
```

Result:

- [joint_student_temporal_v1.json](/opt/tiger/hand/experiments/generated/joint_student_temporal_v1.json)
- scratch sequence accuracy mean: `0.4103`
- teacher-guided sequence accuracy mean: `0.3590`
- scratch window accuracy mean: `0.5023`
- teacher-guided window accuracy mean: `0.4392`

Interpretation:

- under the current MLP student design, temporal-HL regression is a net
  negative signal

### 2. Lower teacher weight

Command:

```bash
python tools/train_joint_student.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --train-pretrain-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_train_joint_3d.json \
  --train-finetune-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_val_joint_3d.json \
  --test-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_test_joint_3d.json \
  --fractions 1.0 \
  --seeds 0 1 2 \
  --pretrain-epochs 20 \
  --finetune-epochs 20 \
  --hidden-dim 128 \
  --teacher-weight 0.1 \
  --output experiments/generated/joint_student_temporal_w01.json
```

Result:

- [joint_student_temporal_w01.json](/opt/tiger/hand/experiments/generated/joint_student_temporal_w01.json)
- scratch sequence accuracy mean: `0.4103`
- teacher-guided sequence accuracy mean: `0.3333`
- scratch window accuracy mean: `0.5023`
- teacher-guided window accuracy mean: `0.4628`

Interpretation:

- simply shrinking the teacher loss does not fix the regression

### 3. Pretrain only, no teacher regression

Command:

```bash
python tools/train_joint_student.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --train-pretrain-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_train_joint_3d.json \
  --train-finetune-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_val_joint_3d.json \
  --test-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_test_joint_3d.json \
  --fractions 1.0 \
  --seeds 0 1 2 \
  --pretrain-epochs 20 \
  --finetune-epochs 20 \
  --hidden-dim 128 \
  --teacher-weight 0.0 \
  --output experiments/generated/joint_student_pretrain_only.json
```

Result:

- [joint_student_pretrain_only.json](/opt/tiger/hand/experiments/generated/joint_student_pretrain_only.json)
- scratch sequence accuracy mean: `0.4103`
- pretrained sequence accuracy mean: `0.4359`
- scratch window accuracy mean: `0.5023`
- pretrained window accuracy mean: `0.4741`

Interpretation:

- joint-label pretraining itself helps sequence accuracy slightly
- but that gain does not come from temporal-HL teacher regression

### 4. Teacher in pretrain only, disabled in finetune

Command:

```bash
python tools/train_joint_student.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --train-pretrain-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_train_joint_3d.json \
  --train-finetune-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_val_joint_3d.json \
  --test-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_test_joint_3d.json \
  --fractions 1.0 \
  --seeds 0 1 2 \
  --pretrain-epochs 20 \
  --finetune-epochs 20 \
  --hidden-dim 128 \
  --pretrain-teacher-weight 0.5 \
  --finetune-teacher-weight 0.0 \
  --output experiments/generated/joint_student_preteach_only.json
```

Result:

- [joint_student_preteach_only.json](/opt/tiger/hand/experiments/generated/joint_student_preteach_only.json)
- scratch sequence accuracy mean: `0.4103`
- preteach-only sequence accuracy mean: `0.3333`
- scratch window accuracy mean: `0.5023`
- preteach-only window accuracy mean: `0.4561`

Interpretation:

- the negative effect is not limited to finetune-time teacher supervision
- current teacher target appears mismatched to the simple joint MLP student

## Current conclusion

The joint-student line is now executable and informative.

What is supported by current evidence:

- raw-joint pretraining on train labels is mildly useful
- current temporal-HL teacher regression is not yet a winning supervision route
- therefore, the present student architecture should not be promoted as a main
  positive result

What this changes in strategy:

- use the joint-student line as a diagnostic branch, not the mainline claim
- do not spend more cycles on teacher-weight tweaks alone
- if this branch continues, the next meaningful upgrade should change the
  student target or architecture, for example:
  - sequence encoder instead of flattened-window MLP
  - teacher classification / contrastive targets instead of raw feature
    regression
  - class-balanced finetune for the persistent ROM confusions

## Cross-representation protocol transfer

New question:

- does the strongest symbolic protocol change
  "pretrain-only time normalization" also help a raw-joint student?

Implementation note:

- [train_joint_student.py](/opt/tiger/hand/tools/train_joint_student.py) now
  supports split-specific temporal window settings:
  - `--pretrain-window-span-units`
  - `--pretrain-window-step-units`
  - matching finetune/test variants
- for time-normalized joint windows, variable-length frame chunks are resampled
  or padded back to a fixed `window_size`, so the MLP input dimension remains
  stable
- teacher/sample alignment remains keyed by `sample_id`

### Reference raw-joint pretrain-only baseline

Command:

```bash
python tools/train_joint_student.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --train-pretrain-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_train_joint_3d.json \
  --train-finetune-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_val_joint_3d.json \
  --test-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_test_joint_3d.json \
  --fractions 1.0 \
  --seeds 0 1 2 \
  --pretrain-epochs 20 \
  --finetune-epochs 20 \
  --hidden-dim 128 \
  --teacher-weight 0.0 \
  --output experiments/generated/joint_student_pretrain_only_default_v2.json
```

Result:

- [joint_student_pretrain_only_default_v2.json](/opt/tiger/hand/experiments/generated/joint_student_pretrain_only_default_v2.json)
- sequence accuracy:
  `scratch=0.4103`, `pretrain_only=0.4359`

### Pretrain-only time normalization on raw joints

Tried:

- `pretrain-window-span-units=168`, `pretrain-window-step-units=84`
- `pretrain-window-span-units=186`, `pretrain-window-step-units=96`
- `pretrain-window-span-units=222`, `pretrain-window-step-units=114`

Results:

- [joint_student_pretrain_only_pre168_84_v1.json](/opt/tiger/hand/experiments/generated/joint_student_pretrain_only_pre168_84_v1.json)
  -> sequence accuracy `0.4103`, window accuracy `0.5248`
- [joint_student_pretrain_only_pre186_96_v1.json](/opt/tiger/hand/experiments/generated/joint_student_pretrain_only_pre186_96_v1.json)
  -> sequence accuracy `0.3846`, window accuracy `0.4730`
- [joint_student_pretrain_only_pre222_114_v1.json](/opt/tiger/hand/experiments/generated/joint_student_pretrain_only_pre222_114_v1.json)
  -> sequence accuracy `0.4103`, window accuracy `0.5191`

Interpretation:

- unlike the symbolic route, pretrain-only time normalization does not improve
  sequence-level raw-joint performance in the current flattened-window MLP
  student
- the `168/84` and `222/114` settings mildly improve window accuracy, but that
  does not translate into better sequence decisions
- therefore, the new symbolic protocol should currently be viewed as a
  representation-specific win, not yet a representation-agnostic training
  protocol

Practical conclusion:

- do not keep pushing the current raw-joint MLP line
- if cross-representation validation is revisited, the next change should be an
  architectural one, such as a sequence encoder over joint trajectories rather
  than more protocol tuning

## Joint sequence encoder branch

New branch goal:

- test whether pretrain-only time normalization transfers once the raw-joint
  student uses an actual sequence encoder rather than a flattened-window MLP

Script:

- [train_joint_sequence_student.py](/opt/tiger/hand/tools/train_joint_sequence_student.py)

Design:

- input is a `window_size x frame_feature_dim` joint sequence
- a lightweight bidirectional GRU encoder models temporal dynamics
- pretraining transfers the input projection and recurrent encoder, then
  finetunes a new classifier head on ROM labels

### Reference sequence-encoder baseline

Command:

```bash
python tools/train_joint_sequence_student.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --train-pretrain-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_train_joint_3d.json \
  --train-finetune-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_val_joint_3d.json \
  --test-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_test_joint_3d.json \
  --fractions 1.0 \
  --seeds 0 1 2 \
  --pretrain-epochs 20 \
  --finetune-epochs 20 \
  --hidden-dim 128 \
  --output experiments/generated/joint_sequence_student_default_v1.json
```

Result:

- [joint_sequence_student_default_v1.json](/opt/tiger/hand/experiments/generated/joint_sequence_student_default_v1.json)
- sequence accuracy:
  `scratch=0.6410`, `pretrained=0.7949`
- window accuracy:
  `scratch=0.6667`, `pretrained=0.7613`

Interpretation:

- this is a much better raw-joint baseline than the flattened-window MLP branch
- temporal modeling at the architecture level matters

### Pretrain-only time normalization on the sequence encoder

Tried:

- `pretrain-window-span-units=168`, `pretrain-window-step-units=84`
- `pretrain-window-span-units=186`, `pretrain-window-step-units=96`
- `pretrain-window-span-units=222`, `pretrain-window-step-units=114`

Results:

- [joint_sequence_student_pre168_84_v1.json](/opt/tiger/hand/experiments/generated/joint_sequence_student_pre168_84_v1.json)
  -> pretrained sequence accuracy `0.8205`
- [joint_sequence_student_pre186_96_v1.json](/opt/tiger/hand/experiments/generated/joint_sequence_student_pre186_96_v1.json)
  -> pretrained sequence accuracy `0.8718`
- [joint_sequence_student_pre222_114_v1.json](/opt/tiger/hand/experiments/generated/joint_sequence_student_pre222_114_v1.json)
  -> pretrained sequence accuracy `0.6923`

Interpretation:

- unlike the raw-joint MLP branch, the sequence encoder does benefit from
  pretrain-only time normalization
- the gain is not identical to the symbolic best setting:
  - symbolic best pretrain-only setting: `168/84`
  - joint-sequence best pretrain-only setting: `186/96`
- this suggests the protocol improvement does transfer across representations,
  but its optimum depends on the downstream model family

Cross-representation conclusion:

- the strongest current evidence is now:
  - symbolic branch: `pretrain-only 168/84` is best
  - joint sequence branch: `pretrain-only 186/96` is best
- therefore, pretrain-only time normalization is no longer just a
  representation-specific symbolic trick
- it behaves like a generally useful pretraining protocol, with
  representation-dependent calibration of the time span

## Unified protocol comparison

Compact table artifact:

## Learned-token control baseline

New question:

- if we replace Temporal HL with a lightweight learned discrete tokenizer over
  raw joint windows, do we get comparable gains just from discretization?

Script:

- [train_joint_token_baseline.py](/opt/tiger/hand/tools/train_joint_token_baseline.py)

Design:

- fit a `MiniBatchKMeans` codebook on pretrain joint-frame features only
- convert each window into a token sequence by nearest-center assignment
- represent the window with token histogram, bigram histogram, and simple
  token-tempo statistics
- run the same pretrain -> finetune classifier protocol used elsewhere

This is intentionally lightweight. It is a control baseline, not a new mainline
model family.

### Smoke run

Command:

```bash
python tools/train_joint_token_baseline.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --train-pretrain-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_train_joint_3d.json \
  --train-finetune-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_val_joint_3d.json \
  --test-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_test_joint_3d.json \
  --fractions 1.0 \
  --seeds 0 \
  --num-tokens 16 \
  --max-codebook-frames 20000 \
  --pretrain-epochs 3 \
  --finetune-epochs 3 \
  --output experiments/generated/joint_token_baseline_smoke.json
```

Result:

- [joint_token_baseline_smoke.json](/opt/tiger/hand/experiments/generated/joint_token_baseline_smoke.json)
- scratch sequence accuracy: `0.0769`
- pretrained sequence accuracy: `0.3077`

Interpretation:

- the learned-token pipeline is executable end to end
- the smoke run is only a sanity check, not a meaningful comparison point

### Main control on the strongest joint-sequence protocol

Reference protocol:

- pretrain-only time normalization
- `pretrain-window-span-units=186`
- `pretrain-window-step-units=96`

#### 32-token codebook

Command:

```bash
python tools/train_joint_token_baseline.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --train-pretrain-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_train_joint_3d.json \
  --train-finetune-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_val_joint_3d.json \
  --test-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_test_joint_3d.json \
  --fractions 1.0 0.5 \
  --seeds 0 1 2 \
  --num-tokens 32 \
  --max-codebook-frames 100000 \
  --pretrain-window-span-units 186 \
  --pretrain-window-step-units 96 \
  --pretrain-epochs 120 \
  --finetune-epochs 120 \
  --output experiments/generated/joint_token_baseline_pre186_96_v1.json
```

Result:

- [joint_token_baseline_pre186_96_v1.json](/opt/tiger/hand/experiments/generated/joint_token_baseline_pre186_96_v1.json)
- fraction `1.0`: scratch `0.5385`, pretrained `0.5897`
- fraction `0.5`: scratch `0.4103`, pretrained `0.5385`

#### 64-token codebook

Command:

```bash
python tools/train_joint_token_baseline.py \
  --train-pretrain-json experiments/generated/temporal_hl_train.json \
  --train-finetune-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --train-pretrain-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_train_joint_3d.json \
  --train-finetune-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_val_joint_3d.json \
  --test-joint-json /opt/tiger/InterHand/annotations/machine_annot/InterHand2.6M_test_joint_3d.json \
  --fractions 1.0 0.5 \
  --seeds 0 1 2 \
  --num-tokens 64 \
  --max-codebook-frames 100000 \
  --pretrain-window-span-units 186 \
  --pretrain-window-step-units 96 \
  --pretrain-epochs 120 \
  --finetune-epochs 120 \
  --output experiments/generated/joint_token_baseline_pre186_96_tok64_v1.json
```

Result:

- [joint_token_baseline_pre186_96_tok64_v1.json](/opt/tiger/hand/experiments/generated/joint_token_baseline_pre186_96_tok64_v1.json)
- fraction `1.0`: scratch `0.7436`, pretrained `0.6923`
- fraction `0.5`: scratch `0.6410`, pretrained `0.5641`

Compact comparison artifact:

- [joint_token_control_summary.json](/opt/tiger/hand/experiments/generated/joint_token_control_summary.json)

Key comparison against the current raw-joint mainline:

- joint sequence encoder with the same pretrain-only `186/96` protocol:
  - fraction `1.0` pretrained sequence accuracy `0.8718`
  - fraction `0.5` pretrained sequence accuracy `0.7179`
- learned-token control:
  - 32 tokens: `0.5897` / `0.5385`
  - 64 tokens: `0.6923` / `0.5641`

Interpretation:

- a lightweight learned discrete tokenizer does not match the stronger raw-joint
  sequence encoder under the same protocol
- increasing codebook size from `32` to `64` does not close the gap
- pretraining is unstable in the learned-token control, with `64` tokens even
  showing `scratch > pretrained`
- this supports the claim that Temporal HL gains are not explained by
  discretization alone
- for now, the learned-token line is a control baseline, not a candidate
  replacement for the mainline representation

- [protocol_table.json](/opt/tiger/hand/experiments/generated/protocol_table.json)
- [protocol_matrix.json](/opt/tiger/hand/experiments/generated/protocol_matrix.json)

Joint-sequence branch summary at fraction `1.0`:

- `default`
  - `scratch_seq=0.6410`
  - `pretrained_seq=0.7949`
- `all_split_norm_186_96`
  - `scratch_seq=0.6154`
  - `pretrained_seq=0.7179`
- `pretrain_only_norm_186_96`
  - `scratch_seq=0.6410`
  - `pretrained_seq=0.8718`

Interpretation:

- for the joint sequence encoder, the same high-level pattern holds as in the
  symbolic branch
- normalizing all splits hurts both scratch and pretrained performance
- the useful effect again comes from pretrain-only normalization
- however, the best calibrated temporal span for the joint sequence model is
  different from the symbolic best setting

From the full protocol matrix:

- at fraction `0.5`, joint-sequence `pretrain_only_norm_186_96` improves
  pretrained sequence accuracy from `0.5897` to `0.7179`
- at fraction `1.0`, it improves from `0.7949` to `0.8718`
- joint-sequence `all_split_norm_186_96` is consistently worse than the
  pretrain-only variant, despite using the same temporal span

## Summary bundle

Single-file bundle artifact:

- [experiment_summary_bundle.json](/opt/tiger/hand/experiments/generated/experiment_summary_bundle.json)

Current joint-sequence gain summary from the bundle:

- fraction `0.5`: `+0.1282`
- fraction `1.0`: `+0.0769`

Practical use:

- this bundle is the current compact handoff artifact that unifies symbolic,
  joint-sequence, protocol, slice, and error-frontier evidence
- table-friendly exports are available under
  [summary_tables](/opt/tiger/hand/experiments/generated/summary_tables)

Joint-sequence slice comparison artifact:

- [joint_sequence_slice_compare_v1.json](/opt/tiger/hand/experiments/generated/joint_sequence_slice_compare_v1.json)

At fraction `1.0`, the main pretrained slice gains are:

- `wrist_rom`: `0.5000 -> 0.8333`
- `right`: `0.7333 -> 0.8667`
- `interaction`: `0.7778 -> 0.8889`

while:

- `finger_occlusion`, `left`, `occlusion`, `no_occlusion`, `touching` remain
  unchanged

Interpretation:

- on the joint sequence branch, pretrain-only normalization helps most on wrist
  and right-hand interaction-heavy cases
- its gain profile is therefore related to, but not identical with, the
  strongest symbolic slice improvements
