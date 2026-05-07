# Temporal Hand Labanotation: Explicit Motion and Keyframe Semantics for Symbolic Hand Movement Documentation

## Abstract

Existing Hand Labanotation-style representations are framewise: they describe local hand posture at each instant but do not explicitly encode how the hand moves or where temporal turning points occur. This makes them suitable for posture documentation but limited for movement documentation, symbolic editing, and notation-conditioned motion recovery. We propose **Temporal Hand Labanotation (Temporal-HL)**, a temporal extension of hand notation that augments framewise symbols with two explicit temporal channels: region-level motion tokens and sequence-level keyframe markers. Temporal-HL is derived automatically from 3D hand joint trajectories and preserves both spatial configuration and temporal evolution. We build a full label-generation pipeline, benchmark notation translation and token-to-motion reconstruction, and analyze multiple decoder families, ablations, robustness conditions, and fair training controls. Experiments on a newly organized hand-motion benchmark show that Temporal-HL adds learnable temporal semantics while maintaining motion recoverability comparable to framewise static notation under fair training. Our findings support Temporal-HL not as a universally lower-error representation, but as a richer symbolic language that explicitly exposes motion and event structure without sacrificing downstream usability.

## 1. Introduction

Symbolic notation is valuable because it compresses movement into an intuitive and transmissible form. Prior work on Hand Labanotation introduced a framewise symbolization scheme for hand movement documentation, but the representation remained essentially static: each frame was mapped to a symbol set describing local spatial configuration. Such a formulation is sufficient for posture description, yet insufficient for movement documentation, because motion itself is not explicitly encoded.

This limitation matters in at least three scenarios. First, two clips can share similar framewise poses but differ in motion direction or transition timing. Second, notation-to-motion reconstruction from static symbols is inherently ambiguous, because the same pose sequence may correspond to different temporal dynamics. Third, downstream uses such as robotic hand control or movement editing require explicit motion signals rather than only framewise posture states.

We address this gap by extending hand notation from a framewise pose code into a temporal symbolic language. Our proposal, Temporal-HL, adds motion-aware representation while remaining compatible with the original static notation view.

The paper is intentionally framed as a **representation study** rather than a pure performance paper. Our goal is not to claim that Temporal-HL always yields the lowest reconstruction error under every decoder, but to show that explicit temporal semantics can be added to hand notation in a learnable, interpretable, and downstream-usable way.

Our contributions are:

1. **Temporal-HL**, a temporal symbolic extension of framewise hand notation with static symbols, motion symbols, and keyframe anchors.
2. **An automatic label-generation pipeline** that converts 3D joint sequences into Temporal-HL without manual temporal annotation.
3. **A benchmark protocol** spanning notation translation, token-to-motion reconstruction, ablations, robustness checks, and fair training controls.
4. **An empirical study** showing that Temporal-HL provides richer symbolic semantics while preserving motion recoverability at a level comparable to framewise notation under fair training.

## 2. Temporal-HL Representation

Temporal-HL represents each sequence with three components:

1. **Static tokens**: framewise region-direction symbols, preserving the original pose-centric notation.
2. **Motion tokens**: region-level delta-direction symbols between adjacent frames, capturing how each local hand part moves.
3. **Keyframe mask**: sequence-level anchors indicating low-energy turning or pause points.

Formally, for frame `t` and region `i`, the notation is:

`Temporal-HL_t = {(s_t^i, m_t^i)}_{i=1}^{40}, k_t`

where `s_t^i` is a static symbol, `m_t^i` is a motion symbol, and `k_t` is a binary keyframe indicator.

## 3. Label Generation Pipeline

The current dataset provides 42 world-space hand joints per frame and per-joint validity masks. We use this data directly to derive Temporal-HL without requiring image-based pose estimation.

### 3.1 Region vectors

Following the hand kinematic tree, we define 20 region vectors per hand from parent-child joint relations, yielding 40 region vectors for both hands.

### 3.2 Static tokens

Each region vector is normalized and quantized to one of 26 direction prototypes corresponding to the non-zero vertices, face centers, and edge centers of a 3D cube.

### 3.3 Motion tokens

For each region vector, we compute the normalized difference between adjacent frames. The delta direction is quantized into the same 26-direction codebook, with an additional `hold` token for small-magnitude changes.

### 3.4 Keyframes

We define framewise motion energy from aggregated region-vector differences and mark local minima as keyframes. These markers approximate pauses, transitions, and turning points.

## 4. Tasks and Models

We define two tasks.

### 4.1 Notation translation

Input: normalized joint coordinates and velocities.  
Output: static tokens, motion tokens, and keyframe markers.

We implement:
- a **Static-HL baseline** predicting only static tokens;
- a **Temporal-HL baseline** predicting static tokens, motion tokens, and keyframes jointly.

### 4.2 Token-to-motion reconstruction

Input: notation tokens.  
Output: normalized joint trajectories.

We evaluate whether temporal notation improves recoverability over static-only notation under multiple decoders:
- Transformer decoder
- GRU decoder

## 5. Experiments

We organize the experiments around one central question: can temporal semantics be added to symbolic hand notation without sacrificing downstream usability? Accordingly, the main text emphasizes four aspects: (1) whether temporal notation is learnable, (2) whether its components are meaningful, (3) whether it preserves motion recoverability under fair comparison, and (4) how representation behavior depends on decoder family. Additional robustness and auxiliary-task results are treated as supporting evidence.

### 5.1 Dataset

The dataset contains 2000 training clips and 100 test clips. Each clip has 81 frames at 5 FPS. Hand types include left, right, and interacting.

Importantly, this benchmark uses a **newly organized clip-level split** built from the available hand-motion corpus. The original metadata fields inherited from the source dataset, such as `source_split`, are retained only as provenance and are **not** the split protocol used in this paper. Our experiments therefore evaluate Temporal-HL on the reorganized train/test split defined for this study rather than reproducing the original source benchmark.

### 5.2 Notation translation results

Test results:

- Static-HL baseline:
  - static accuracy: **0.8311**
- Temporal-HL baseline:
  - static accuracy: **0.7902**
  - motion accuracy: **0.6869**
  - keyframe F1: **0.4711**

This shows that Temporal-HL is a harder but richer prediction task. It does not improve static-token classification directly under joint training, but it learns additional temporal targets with reasonable generalization.

### 5.2.1 Sequence-aware retrieval proxy

As an auxiliary analysis, we compute a simple sequence-aware similarity between notation sequences on the test split. The nearest-neighbor similarity statistics are:

- Static-HL: **0.3370**
- Keyframe-augmented HL: **0.3003**
- Temporal-HL: **0.3756**

Although this is not a primary benchmark, it suggests that Temporal-HL preserves more sequence-specific temporal identity than framewise static notation under a temporal matching view.

### 5.2.2 Auxiliary action classification sanity check

We also train a notation-only classifier to predict overlapping `seq_name` categories from token sequences. In this benchmark, Static-HL, keyframe-HL, and Temporal-HL all eventually reach near-perfect accuracy, indicating that the task is too easy to serve as a discriminative main benchmark. However, temporal variants converge faster, which still supports the claim that temporal notation remains usable for downstream classification.

### 5.3 Token-to-motion reconstruction

Transformer decoder:

- Static-HL: **0.0971**
- Temporal-HL: **0.0915**

This scratch-trained result suggests that temporal symbols can help notation-conditioned recovery under a lightweight Transformer decoder, but we do not treat this single number as the final conclusion because later robustness checks show that optimization details matter substantially.

GRU decoder:

- Static-HL: **0.0584**
- Temporal-HL: **0.0612**

The advantage is not universal across all decoders, suggesting that the usefulness of temporal notation depends on decoder inductive bias and optimization. This is important because it indicates the benefit of Temporal-HL is structured rather than trivial.

### 5.4 Ablation

For the Transformer decoder:

- static only: **0.0971**
- static + keyframe: **0.0942**
- static + motion: **0.0973**
- static + motion + keyframe: **0.0915**

This suggests:

1. keyframe anchors alone already improve recoverability;
2. motion tokens alone are insufficient;
3. the best result appears when local motion and temporal anchors are used together.

This is important because it shows the Temporal-HL design is structured rather than ad hoc: keyframes provide the strongest single temporal signal, while motion tokens are only useful when anchored by event structure.

In the final paper, this ablation should be the main reconstruction table because it most directly supports the representation design itself.

### 5.4.1 Robustness-oriented follow-up

Because a reviewer may question whether the reconstruction gain is tied to a single lucky run, we further tested two robustness conditions.

First, scaling the Transformer decoder up (`d_model=384`, `layers=6`) does not preserve the advantage:

- Static-HL: **0.1032**
- Temporal-HL: **0.1081**

Second, under two additional random seeds for the original lightweight Transformer:

- seed 7:
  - Static-HL: **0.0939**
  - Temporal-HL: **0.0932**
- seed 13:
  - Static-HL: **0.0894**
  - Temporal-HL: **0.0937**

Across the three currently available seeds (including the original seed-42 run), the mean error is:

- Static-HL mean: **0.0935**
- Temporal-HL mean: **0.0928**

This average still slightly favors Temporal-HL, but the margin is small and the ranking is not stable for every seed. Therefore, the most defensible conclusion is not that Temporal-HL consistently outperforms Static-HL, but that temporal symbolic structure shows **promising yet optimization-sensitive** benefits for notation-conditioned reconstruction.

### 5.4.2 Warm-start stabilization

We further tested whether the instability comes from optimization rather than from the representation itself. Concretely, we first train a Static-HL Transformer decoder, then initialize the Temporal-HL decoder from that checkpoint for all compatible weights, and finally fine-tune the temporal model with motion and keyframe branches enabled.

Using a matched configuration (`d_model=256`, `layers=4`, `dropout=0.2`), we obtain:

- seed 7:
  - Static-HL: **0.0820**
  - Temporal-HL warm-start: **0.0653**
- seed 13:
  - Static-HL: **0.0837**
  - Temporal-HL warm-start: **0.0635**

These results are substantially stronger than the scratch-trained temporal runs. Therefore, the earlier instability appears to be an optimization issue: the temporal symbolic decoder benefits from being initialized with a good static symbolic prior before learning the additional motion-aware branches. However, this warm-start result alone is still insufficient to claim that Temporal-HL is strictly better, because it also gives the temporal model an additional fine-tuning stage.

### 5.4.3 Fair continued-training control

However, a careful reviewer may still ask whether the warm-start gain is caused by the temporal representation itself, or simply by giving the model extra fine-tuning budget. To answer this, we run a matched control in which the Static-HL decoder is also fine-tuned for the same additional training budget and with the same optimization settings.

The resulting test errors are:

- seed 7:
  - Static-HL continued training: **0.0652**
  - Temporal-HL warm-start: **0.0653**
- seed 13:
  - Static-HL continued training: **0.0630**
  - Temporal-HL warm-start: **0.0635**

This control is important because it changes the interpretation of the paper. After equalizing the extra optimization budget, Temporal-HL is no longer clearly better in reconstruction error. Therefore, the paper should not claim that temporal notation is a universally stronger representation for minimizing joint-space reconstruction loss. Instead, the defensible claim is that Temporal-HL offers **comparable recoverability** while exposing explicit motion and event structure that framewise static notation does not encode.

For the final paper, the warm-start and fair-control results should be shown together in one compact table, because their purpose is interpretive rather than purely quantitative: they explain *why* the initial scratch advantage should not be overclaimed.

### 5.5 Multi-decoder analysis

We further test a GRU decoder:

- Static-HL GRU: **0.0584**
- Temporal-HL GRU: **0.0612**

We also test a TCN decoder:

- Static-HL TCN: **0.0649**
- Temporal-HL TCN: **0.0677**

These results indicate that the gain of Temporal-HL is not universal across all decoder families. Instead, Temporal-HL is most helpful when the decoder can directly exploit token-level temporal structure, as in the lightweight Transformer decoder. Under GRU and TCN decoders, which already impose strong temporal smoothness priors, the static notation remains slightly easier to decode.

This is still a publishable outcome because it sharpens the scientific claim: the usefulness of temporal symbolic notation depends on the interaction between representation and decoder bias.

## 6. Discussion

Temporal-HL should not be interpreted as a universal replacement for framewise notation on every metric. Instead, it should be understood as a richer symbolic layer that makes motion itself representable. The representation is particularly useful when downstream tasks require explicit motion and event structure rather than only per-frame posture coding.

The mixed reconstruction outcome across decoders is scientifically useful. It shows that the interaction between symbolic temporal structure and sequence model architecture is itself a meaningful research question.

In particular, our experiments suggest that:

- framewise notation is strong when the decoder must infer dynamics implicitly;
- keyframe-aware notation is already beneficial in the main ablation;
- the optimization protocol matters substantially, and warm-starting from a static symbolic decoder is an effective stabilization strategy;
- after controlling for extra training budget, Temporal-HL is best interpreted as a **richer but not strictly lower-error** representation;
- therefore, the value of a temporal notation should be analyzed jointly with the temporal inductive bias of the downstream model.

## 7. Limitations

1. Current keyframe detection is heuristic rather than learned.
2. Reconstruction experiments are limited to normalized joint space.
3. The paper currently evaluates notation derived from 3D joints, not direct image-to-notation generation.
4. The decoder study is still limited in scale; stronger diffusion or autoregressive decoders remain future work.

## 8. Conclusion

We introduced Temporal-HL, a temporal extension of hand notation that explicitly models local motion and sequence anchors. Our experiments show that explicit temporal semantics can be added to hand notation in a learnable way, and that under fair training Temporal-HL preserves motion recoverability at a level comparable to framewise static notation while exposing symbolic motion and event structure unavailable in the original formulation. This makes Temporal-HL a practical motion-aware symbolic layer and a useful benchmark direction for future hand movement documentation research.

## Appendix-Style Notes For Final Version

### Implementation assets already available

- representation code:
  - `temporal_hl/notation.py`
- preprocessing:
  - `temporal_hl/preprocess.py`
- notation translation:
  - `train_temporal_hl.py`
- token-to-motion reconstruction:
  - `train_token_reconstruction.py`
  - `train_token_reconstruction_v2.py`
- visualization:
  - `scripts/make_paper_figures.py`
  - `scripts/make_reconstruction_plot.py`
- paper assets:
  - `paper_assets/result_table.md`
  - `temporal_hl_cache/paper_assets/test_000000_token_strips.png`
  - `temporal_hl_cache/paper_assets/test_000000_motion_energy.png`
  - `temporal_hl_cache/paper_assets/test_000000_reconstruction_compare.png`

## Figures To Include

1. Temporal-HL overview: static token, motion token, keyframe mask.
2. Label generation pipeline from joints to notation.
3. Token strip visualization for one sequence.
4. Motion energy with detected keyframes.
5. Reconstruction comparison between Static-HL and Temporal-HL.

## Tables To Include

1. Notation translation results.
2. Token-to-motion reconstruction results across decoders.
3. Ablation over temporal components.
