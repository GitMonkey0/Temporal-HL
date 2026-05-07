## Main Table 1. Notation Translation

| Method | Static Acc ↑ | Motion Acc ↑ | Keyframe F1 ↑ |
|---|---:|---:|---:|
| Static-HL | **0.8311** | - | - |
| Temporal-HL | 0.7902 | 0.6869 | 0.4711 |

Interpretation:
- Temporal-HL is a richer and harder prediction task.
- The goal is not to outperform Static-HL on static-token accuracy, but to add explicit temporal semantics that remain learnable.

## Main Table 2. Reconstruction Ablation Under Scratch Training

| Representation | Coord L1 ↓ |
|---|---:|
| Static only | 0.0971 |
| Static + motion | 0.0973 |
| Static + keyframe | 0.0942 |
| Static + motion + keyframe | **0.0915** |

Interpretation:
- Keyframes are the strongest single temporal component.
- Motion tokens help only when anchored by event structure.

## Main Table 3. Fair Recoverability Comparison

| Training Protocol | Seed | Static-HL ↓ | Temporal-HL ↓ |
|---|---:|---:|---:|
| Scratch | 42 | 0.0971 | **0.0915** |
| Warm-start | 7 | 0.0820 | **0.0653** |
| Warm-start | 13 | 0.0837 | **0.0635** |
| Fair continued-training control | 7 | **0.0652** | 0.0653 |
| Fair continued-training control | 13 | **0.0630** | 0.0635 |

Interpretation:
- Temporal-HL can be optimized to strong recoverability.
- Under a fair extra-training control, its advantage in reconstruction error disappears.
- Therefore the paper should claim **comparable recoverability with richer symbolic semantics**, not universal lower error.

## Main Table 4. Decoder Family Analysis

| Decoder | Static-HL ↓ | Temporal-HL ↓ |
|---|---:|---:|
| Transformer (scratch) | 0.0971 | **0.0915** |
| GRU | **0.0584** | 0.0612 |
| TCN | **0.0649** | 0.0677 |

Interpretation:
- The usefulness of temporal notation depends on decoder inductive bias.
- Temporal-HL is best presented as a representation study rather than a universal performance win.
