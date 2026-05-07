## Appendix Table A1. Larger Transformer Robustness Check

| Decoder | Static-HL ↓ | Temporal-HL ↓ |
|---|---:|---:|
| Transformer-Large (`d=384, L=6`) | **0.1032** | 0.1081 |

Interpretation:
- Scaling the same decoder family does not preserve the scratch advantage.

## Appendix Table A2. Multi-Seed Scratch Runs

| Seed | Static-HL ↓ | Temporal-HL ↓ |
|---|---:|---:|
| 42 | 0.0971 | **0.0915** |
| 7 | 0.0939 | **0.0932** |
| 13 | **0.0894** | 0.0937 |
| Mean | 0.0935 | **0.0928** |

Interpretation:
- The scratch advantage is small on average and unstable per seed.

## Appendix Table A3. Sequence-Aware Retrieval Proxy

| Representation | NN Similarity ↑ |
|---|---:|
| Static-HL | 0.3370 |
| Keyframe-HL | 0.3003 |
| Temporal-HL | **0.3756** |

Interpretation:
- Temporal-HL retains more sequence-specific identity under a temporal matching view.
- This remains supporting analysis rather than a main benchmark.

## Appendix Table A4. Action Classification Sanity Check

Qualitative summary:
- Static-HL, keyframe-HL, and Temporal-HL all approach near-perfect accuracy.
- Keyframe-HL and Temporal-HL converge faster.

Interpretation:
- The benchmark is too easy for a main claim.
- It is kept only as a learnability sanity check.
