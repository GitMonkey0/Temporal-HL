# Temporal-HL Experimental Summary

## Setup

- Dataset: current `temporal_hl_cache/data/` split
  - train clips: 2000
  - test clips: 100
  - fps: 5
  - frames per clip: 81
- Input supervision:
  - 42 hand joints in world coordinates
  - per-joint valid mask
  - hand types: `left`, `right`, `interacting`

## Representation

`Temporal-HL` extends framewise Hand Labanotation with:

1. `Static tokens`
   - 40 region vectors per frame
   - 26 directional categories
2. `Motion tokens`
   - delta direction between adjacent frames
   - 26 directional categories + 1 hold token
3. `Keyframe mask`
   - local minima over motion energy

## Experiment A: Token Translation

### A1. Static-only baseline

Command:

```bash
python train_temporal_hl.py \
  --manifest temporal_hl_cache/artifacts/temporal_hl_smoke/manifest.json \
  --epochs 12 --batch-size 64 --lr 5e-4 \
  --mode static \
  --save-dir temporal_hl_cache/runs/static_baseline
```

Test result:

- static accuracy: `0.8311`

### A2. Temporal-HL baseline

Command:

```bash
python train_temporal_hl.py \
  --manifest temporal_hl_cache/artifacts/temporal_hl_smoke/manifest.json \
  --epochs 12 --batch-size 64 --lr 5e-4 \
  --mode temporal \
  --motion-weight 0.5 \
  --keyframe-weight 0.2 \
  --keyframe-pos-weight 8.0 \
  --save-dir temporal_hl_cache/runs/temporal_tuned
```

Test result:

- static accuracy: `0.7902`
- motion accuracy: `0.6869`
- keyframe F1: `0.4711`

Interpretation:

- Temporal-HL does not outperform static-only on the static-token classification objective.
- But it learns two additional sequence-level prediction targets with reasonable generalization.
- This means Temporal-HL is not just a harder version of HL. It is a richer multi-component notation.

## Experiment A.5: Sequence-Aware Retrieval Proxy

We additionally evaluated a simple sequence-aware token similarity proxy on the test split:

- static: `nn_sim_mean = 0.3370`
- keyframe: `nn_sim_mean = 0.3003`
- temporal: `nn_sim_mean = 0.3756`

This is not a core benchmark result, but it suggests that Temporal-HL retains more sequence-specific identity than static-only notation under a temporal similarity measure.

## Experiment A.6: Auxiliary Action Classification Sanity Check

We trained a notation-only classifier to predict overlapping `seq_name` categories from token sequences.

Observed behavior:

- Static-HL, keyframe-HL, and Temporal-HL all eventually reached near-perfect accuracy on this small benchmark.
- Keyframe-HL and Temporal-HL converged faster than Static-HL.

Conclusion:

- This task is too easy to serve as a main benchmark.
- It is still useful as a sanity check that the temporal notation remains learnable and usable for downstream sequence classification.

## Experiment B: Token-to-Motion Reconstruction

This is the more important experiment for the paper.

Question:
Can the notation recover motion better when temporal symbols are included?

### B1. Static token reconstruction

Command:

```bash
python train_token_reconstruction.py \
  --manifest temporal_hl_cache/artifacts/temporal_hl_smoke/manifest.json \
  --epochs 10 --batch-size 64 --lr 5e-4 \
  --mode static \
  --save-dir temporal_hl_cache/runs/token_recon_static
```

Test result:

- coord L1: `0.0971`

### B2. Temporal-HL reconstruction

Command:

```bash
python train_token_reconstruction.py \
  --manifest temporal_hl_cache/artifacts/temporal_hl_smoke/manifest.json \
  --epochs 12 --batch-size 64 --lr 5e-4 \
  --mode temporal \
  --save-dir temporal_hl_cache/runs/token_recon_temporal_kf
```

Test result:

- coord L1: `0.0915`

Relative improvement over static-only:

- `(0.0971 - 0.0915) / 0.0971 = 5.8%`

### B3. Reconstruction ablation

We also observed an important ablation:

- static only: `0.0971`
- static + keyframe: `0.0942`
- static + motion: `0.0973`
- static + motion + keyframe: `0.0915`

Interpretation:

- Keyframe information alone already improves reconstruction.
- Motion tokens alone are not enough.
- The best result appears when local motion changes and keyframe anchors are both present.

This is useful for the paper because it shows the representation is not just “more tokens”, but a structured temporal notation.

### B4. Stronger decoder check

We additionally tested a GRU decoder:

- Static-HL GRU: `0.0584`
- Temporal-HL GRU: `0.0612`

We further tested a TCN decoder:

- Static-HL TCN: `0.0649`
- Temporal-HL TCN: `0.0677`

Interpretation:

- Temporal-HL is **not** universally better under every decoder.
- The current benefit is strongest under the lightweight Transformer decoder.
- Under GRU and TCN, static notation remains slightly better.
- This suggests Temporal-HL is most useful when the model can directly exploit explicit symbolic temporal structure, rather than relying mostly on implicit temporal smoothing.

### B5. Reviewer-oriented robustness checks

To test whether the Transformer gain is tied to one narrow setting, we ran two additional checks.

#### (a) Larger Transformer

Using a larger Transformer decoder (`d_model=384`, `layers=6`):

- Static-HL: `0.1032`
- Temporal-HL: `0.1081`

Interpretation:

- Scaling the same decoder family does **not** preserve the original Temporal-HL advantage.
- The benefit therefore cannot be claimed as a generic “more capacity helps Temporal-HL” result.

#### (b) Multi-seed check on the main Transformer setup

Main setup (`d_model=256`, `layers=4`) additional seeds:

- seed 7:
  - Static-HL: `0.0939`
  - Temporal-HL: `0.0932`
- seed 13:
  - Static-HL: `0.0894`
  - Temporal-HL: `0.0937`

Together with the original seed-42 run:

- Static-HL: `0.0971`, `0.0939`, `0.0894`
- Temporal-HL: `0.0915`, `0.0932`, `0.0937`

Mean over 3 seeds:

- Static-HL mean: `0.0935`
- Temporal-HL mean: `0.0928`

Interpretation:

- The average still slightly favors Temporal-HL.
- However, the margin is small and the per-seed ranking is not stable.
- Reviewer-wise, this means the strongest honest claim is **not** “Temporal-HL consistently outperforms Static-HL”, but rather:
  - Temporal-HL shows promising recoverability gains in some Transformer-style symbolic decoders;
  - the effect is currently sensitive to optimization and decoder bias;
  - robustness remains an open issue.

### B6. Stability repair via warm-start training

We then tested whether the instability came from optimization rather than from the representation itself.

Protocol:

1. Train a Static-HL Transformer decoder first.
2. Initialize the Temporal-HL decoder from the Static-HL checkpoint for all compatible weights.
3. Fine-tune the Temporal-HL decoder with the temporal branches enabled.

Using a matched configuration (`d_model=256`, `layers=4`, `dropout=0.2`, `lr=3e-4` for static pretraining, then warm-start temporal fine-tuning with `lr=2e-4`), we obtained:

- seed 7:
  - Static-HL: `0.0820`
  - Temporal-HL warm-start: `0.0653`
- seed 13:
  - Static-HL: `0.0837`
  - Temporal-HL warm-start: `0.0635`

Interpretation:

- Under a better-structured optimization protocol, Temporal-HL consistently and substantially outperforms Static-HL.
- This strongly suggests that the earlier instability was mainly an optimization issue caused by training the richer temporal representation from scratch.
- Reviewer-wise, the paper can now make a much stronger claim:
  - Temporal-HL is beneficial for notation-conditioned reconstruction;
  - but the temporal decoder should be initialized from a good static symbolic prior.

### B7. Fair continued-training control

We then tested the most important reviewer control: whether the warm-start gain is actually caused by the temporal representation, or simply by giving the model extra fine-tuning budget.

Protocol:

- Start from the same stabilized Static-HL checkpoints.
- Fine-tune **Static-HL itself** for the same extra training budget and with the same optimizer settings used in the warm-start Temporal-HL runs.

Results:

- seed 7:
  - Static-HL continued training: `0.0652`
  - Temporal-HL warm-start: `0.0653`
- seed 13:
  - Static-HL continued training: `0.0630`
  - Temporal-HL warm-start: `0.0635`

Interpretation:

- After controlling for extra training budget, the reconstruction advantage essentially disappears.
- Therefore, the paper should **not** claim that Temporal-HL is a strictly better representation for minimizing reconstruction error.
- The strongest honest conclusion is now:
  - Temporal-HL provides **comparable recoverability** to Static-HL under a fair training protocol,
  - while additionally exposing explicit motion and keyframe structure that Static-HL does not represent.

## Current Paper-Ready Claim

The main empirical claim that is already supported:

> Temporal-HL provides motion-aware symbolic structure with reconstruction performance comparable to Static-HL under a fair training protocol, while enabling explicit motion and keyframe representation unavailable in framewise static notation.

This is the strongest current result because it directly validates the representation itself, not only the prediction model.

## Recommended Paper Framing

### Title direction

- `Temporal Hand Labanotation: Motion-Aware Symbolic Representation for Hand Movement Documentation`
- `From Framewise Hand Symbols to Temporal Hand Notation`

### Contribution list

1. A temporal extension of Hand Labanotation with static, motion, and keyframe components.
2. An automatic pipeline to derive Temporal-HL from 3D hand joint sequences.
3. A benchmark protocol for:
   - notation translation
   - token-to-motion reconstruction
4. Empirical evidence that temporal notation improves motion recoverability under Transformer-style symbolic decoders.

## What Still Needs To Be Added For Submission

1. More ablations
   - static + motion
   - static + keyframe
   - static + motion + keyframe
2. TCN / stronger temporal decoder check
3. Visualization figures
   - token strips
   - reconstructed trajectories
   - keyframe alignment plots
4. Comparison against the original HL formulation on the new dataset
5. Writing package
   - abstract
   - method figure
   - experiment table templates
