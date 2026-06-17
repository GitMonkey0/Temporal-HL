# Verified Related-Work Memo for Temporal HL

Date: 2026-06-09

This file is a research memo for project planning and experiment design.
It is not paper text.

## Goal

Externally verify the neighboring literature around a temporal upgrade of Hand
Labanotation (HL), and convert that scan into concrete positioning constraints.

## Bottom line

The literature does not support a weak continuation such as:

- better frame-wise HL translation
- HL plus a stronger temporal backbone
- generic hand-motion discretization

The strongest still-defensible space is:

- sequence-native symbolic hand representation
- explicit temporal channels in the representation itself
- anatomy-aware factorization
- human readability and editability
- control / repair / local edit / transition reasoning advantages over opaque
  learned tokens

## 1. Direct parent work

### Hand Labanotation

- Ling Li et al., ACM MM 2024.
- Already establishes:
  - a 26-symbol hand-space inventory
  - HLD with more than 4M annotated images
  - MHLFormer for image-to-frame HL translation

Implication:

- A new submission cannot be centered on "better HL recognition".
- A plain temporal encoder over the old label space will look incremental.
- The label space itself has to be upgraded.

## 2. Body-motion notation is active, but not finger-articulation-native

Representative works:

- Cai et al., Neural Computing and Applications 2023:
  automatic Labanotation generation from folk-dance videos.
- Jiang et al., IJCNN 2024:
  motion part-level interpolation and manipulation over symbolic Labanotation.
- Yan et al., Eurographics 2025:
  LabanLab for interactive Labanotation authoring with motion preview.
- Jiang et al., CVPR 2026:
  LaMoGen, which introduces LabanLite as a symbolic bridge for text-to-motion.

Implication:

- "Notation is useful" is already established.
- The open hand-specific space is not generic notation, but
  hand-articulation-specific temporal factorization that body-motion pipelines
  do not expose.

## 3. Sign-language notation is adjacent but should not become our identity

Anchor systems:

- HamNoSys:
  a phonetic transcription system for sign languages.
- SignWriting:
  a writing system for sign languages.

Implication:

- These systems justify symbolic movement description, but they are centered on
  linguistic structure, not generic non-semantic hand articulation.
- If Temporal HL drifts toward sign-language phonology, reviewers can redirect
  the work into the sign-language notation literature immediately.

Safe boundary:

- stay with generic hand articulation
- avoid making semantic sign recognition / production the core identity

## 4. Learned discrete tokenization is the main novelty threat

### Sign-language side

Representative works:

- Abzaliev and Mihalcea, EMNLP 2024:
  unsupervised discrete representations for ASL fingerspelling.
- Walsh et al., 2024:
  a data-driven representation for sign language production via learned
  codebooks.
- Zuo et al., ICCV 2025:
  Signs as Tokens, discretizing continuous signs into body-part token streams.
- Gueuwou et al., ACL 2025:
  SHuBERT, multi-stream clustered sign representation learning.
- Symeonidis-Herzig et al., 2026:
  M3T, discrete multi-modal motion tokens with explicit non-manual channels.

### Hand / HOI side

Representative works:

- Huang et al., CVPR 2025:
  HOIGPT, using tokenized long-sequence hand-object interaction modeling.
- He et al., CVPR 2026:
  TokenHand, using discrete tokens for efficient hand mesh reconstruction.

Implication:

- "We discretize hand motion" is not enough.
- "We tokenize hand motion for sequence modeling" is also not enough.
- The only robust contrast is:
  - explicit symbolic semantics
  - anatomical decomposition
  - local controllability / editability
  - transparent temporal channels such as transition, persistence, and motif
    structure

## 5. Generic temporal hand modeling is also crowded

Representative works:

- Wen et al., CVPR 2023:
  hierarchical temporal transformer for 3D hand pose estimation and action
  recognition.
- Mamedov et al., ECCV 2024:
  HandFormer, using dense 3D hand poses with sparse RGB for action modeling.
- Yu et al., CVPR 2025:
  Dyn-HaMR for recovering 4D interacting hand motion from monocular video.
- Sun et al., ICLR 2026:
  UniHand for diverse controlled 4D hand motion modeling.

Implication:

- "Temporal hand modeling" by itself is not novel enough.
- "Temporal smoothness" and "temporal consistency" are baseline expectations,
  not a main claim.
- Representation-level temporal structure must be central.

## 6. Fast-moving outer frontier

Recent 2026 pressure from the generation / interaction side:

- HandX scales bimanual motion and interaction generation and explicitly argues
  that whole-body models miss finger articulation, contact timing, and
  inter-hand coordination.
- TSHaMo targets text-driven 3D hand motion generation and argues existing work
  either over-focuses on full-body motion or requires explicit object meshes.

Implication:

- The field is moving toward hand-specific temporal structure quickly.
- Inter-hand coordination and transition dynamics are now central enough that a
  new symbolic representation should probably encode them explicitly rather than
  leave them implicit in an encoder.

## 7. Positioning constraints for our project

What looks weak:

- frame-wise HL + temporal transformer
- sequence classification framed as the main win
- generic tokenization / compression story

What still looks strong:

- sequence-native HL rather than frame-native HL
- explicit state / transition split
- transition-conditioned motifs or event channels
- anatomy-aware grouped structure rather than monolithic codes
- editability, repairability, and control-facing locality

## 8. Experiment pressure implied by the literature

To survive reviewer comparison against tokenization and temporal modeling, the
work likely needs evidence for:

- representation benefit beyond frame-wise HL
- benefit beyond a stronger temporal encoder on the old labels
- local edit / repair / control advantages over opaque learned tokens
- hard-slice robustness for occlusion, interaction, and difficult articulation
- transition-aware reasoning, not only state prediction

## 9. Practical conclusion for next-stage work

If the goal is a strong AAAI 2027 submission, the safest mainline is:

- define Temporal HL as a representation upgrade
- make transitions first-class
- treat current-frame-only editing as insufficient
- show why explicit symbolic temporal channels improve controllability and
  structure under interacting-hand settings

Avoid spending the main story budget on:

- flat recognition numbers
- generic sequence backbones
- learned-token-vs-symbolic classification contests
