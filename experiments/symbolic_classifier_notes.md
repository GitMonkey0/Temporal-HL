# Symbolic Classifier Notes

This is an experiment log, not paper text.

## Task definition

- Train windows come from `val` ROM sequences
- Test windows come from overlapping `test` ROM sequences
- Labels are canonicalized for known naming variants:
  - `ROM07_RT_Finger_Occlusions` -> `ROM07_Rt_Finger_Occlusions`
  - `ROM08_LT_Finger_Occlusions` -> `ROM08_Lt_Finger_Occlusions`
- Window predictions are aggregated back to sequence predictions

## Script

- [train_symbolic_classifier.py](/opt/tiger/hand/tools/train_symbolic_classifier.py)

## Important finding

Window accuracy alone was misleading. Temporal HL was already stronger on
window classification, but naive sequence aggregation could hide that gain.
Using `mean_log_prob` aggregation restores the sequence-level benefit.

## Default config result

Command:

```bash
python tools/train_symbolic_classifier.py \
  --train-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --aggregation mean_log_prob \
  --window-size 32 \
  --stride 16 \
  --c-value 4.0 \
  --output experiments/generated/symbolic_classifier_logprob.json
```

Result at full training windows:

- `state_only`: sequence accuracy `0.3846`
- `temporal_hl`: sequence accuracy `0.4615`

## Window-size ablation at `C=4.0`

- `window=16`: state `0.4615`, temporal `0.4615`
- `window=24`: state `0.3846`, temporal `0.4615`
- `window=32`: state `0.3846`, temporal `0.4615`
- `window=48`: state `0.3077`, temporal `0.2308`
- `window=64`: state `0.2308`, temporal `0.2308`

Implication:

- temporal HL helps most at intermediate temporal spans
- longer windows are not automatically better

## `C` sweep at `window=32`, `stride=16`, `mean_log_prob`

- `C=0.25`: state `0.0769`, temporal `0.2308`
- `C=0.5`: state `0.0769`, temporal `0.2308`
- `C=1.0`: state `0.2308`, temporal `0.2308`
- `C=2.0`: state `0.2308`, temporal `0.3077`
- `C=4.0`: state `0.3846`, temporal `0.4615`
- `C=8.0`: state `0.4615`, temporal `0.5385`
- `C=16.0`: state `0.5385`, temporal `0.6923`

Implication:

- temporal HL is not only better at one lucky regularization point
- the gap stays positive across a useful `C` range and widens at stronger fits

## Current strongest lightweight result

Command:

```bash
python tools/train_symbolic_classifier.py \
  --train-json experiments/generated/temporal_hl_val.json \
  --test-json experiments/generated/temporal_hl_test.json \
  --aggregation mean_log_prob \
  --window-size 32 \
  --stride 16 \
  --c-value 16.0 \
  --output experiments/generated/symbolic_classifier_c16.json
```

Learning-curve result:

- Fraction `0.25`: state `0.3538`, temporal `0.3846`
- Fraction `0.5`: state `0.4615`, temporal `0.5385`
- Fraction `1.0`: state `0.5385`, temporal `0.6923`

Window accuracies under the same config:

- Fraction `0.25`: state `0.5189`, temporal `0.6155`
- Fraction `0.5`: state `0.5858`, temporal `0.6878`
- Fraction `1.0`: state `0.6216`, temporal `0.7264`

## Immediate implication

The trainable baseline now supports two stronger claims:

- temporal HL improves window-level discriminability
- with the right aggregation and regularization, that gain transfers to
  sequence-level classification and sample efficiency
