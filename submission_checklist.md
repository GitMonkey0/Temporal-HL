# Submission Checklist

## Already Done

- Temporal-HL representation defined
- Automatic label generation pipeline implemented
- Notation translation baselines trained
- Token-to-motion reconstruction benchmarks trained
- Multi-decoder analysis completed
- Core ablation completed
- Paper figures generated
- First full paper draft written

## Strong Evidence Already Available

- Transformer decoder:
  - Static-HL: `0.0971`
  - Temporal-HL: `0.0915`
  - Relative improvement: `5.8%`
- Warm-start stabilized Transformer:
  - seed 7:
    - Static-HL: `0.0820`
    - Temporal-HL warm-start: `0.0653`
  - seed 13:
    - Static-HL: `0.0837`
    - Temporal-HL warm-start: `0.0635`
- Fair continued-training control:
  - seed 7:
    - Static-HL continued training: `0.0652`
    - Temporal-HL warm-start: `0.0653`
  - seed 13:
    - Static-HL continued training: `0.0630`
    - Temporal-HL warm-start: `0.0635`
- Ablation:
  - static only: `0.0971`
  - static + keyframe: `0.0942`
  - static + motion: `0.0973`
  - full temporal: `0.0915`

## What Makes The Paper Credible

- The result is not overstated.
- The decoder dependency is explicitly analyzed.
- Keyframe contribution is isolated.
- Negative result on GRU/TCN is preserved instead of hidden.
- The main stability concern is understood and documented.
- The fair-control result prevents overclaiming and makes the paper more defensible.

## Still Recommended Before Submission

### Must-have polishing

- Rewrite the draft in actual target venue style
- Convert main tables and appendix tables to publication-quality tables
- Add method overview figure
- Add 1-2 more qualitative reconstruction figures

### Table organization

- Main text:
  - notation translation
  - reconstruction ablation
  - fair recoverability comparison
  - decoder family analysis
- Appendix:
  - large Transformer
  - scratch multi-seed
  - retrieval proxy
  - action classification sanity check

### Nice-to-have experiments

- Learned keyframe detector instead of heuristic one
- Original HL vs Temporal-HL under a shared decoder with matched capacity
- A downstream task that directly uses motion/keyframe semantics rather than only reconstruction loss

## Current Maturity Estimate

- Workshop-ready: yes
- Borderline top-tier submission-ready: not ideal
- Comfortable top-tier submission-ready: no

## Most Honest Positioning

This should be positioned as a **representation paper** with the main claim:

> Temporal-HL adds explicit motion and keyframe semantics to framewise hand notation, while preserving comparable motion recoverability under fair training. The main value is richer symbolic expressiveness, not strictly lower reconstruction error.
