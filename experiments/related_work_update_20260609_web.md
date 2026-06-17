# Related Work Update from Web Scout

Date: 2026-06-09

This file is a research memo for experiment planning. It is not paper text.

## Purpose

This addendum updates the local related-work scan with externally verified
2024-2026 papers and systems that are closest to a temporal upgrade of Hand
Labanotation.

## 1. Direct baseline: Hand Labanotation already owns frame-wise symbolic hand notation

- Ling Li et al., "Translating Motion to Notation: Hand Labanotation for
  Intuitive and Comprehensive Hand Movement Documentation", ACM MM 2024.
- The released story is already:
  - a 26-symbol hand-space inventory
  - HLD with more than 4M annotated images
  - MHLFormer for image-to-frame HL translation

Implication:

- A new submission cannot center the story on "better HL recognition".
- A plain "HL + temporal encoder" upgrade is weak and exposed to incremental
  criticism.
- The representation itself must change if we want a strong continuation.

## 2. Automatic Labanotation is active, but mostly full-body and pipeline-centric

Representative works:

- Xingquan Cai et al., "Automatic generation of Labanotation based on human
  pose estimation in folk dance videos", Neural Computing and Applications,
  2023.
- Junkun Jiang et al., "Motion Part-Level Interpolation and Manipulation over
  Automatic Symbolic Labanotation Annotation", IJCNN 2024.
- Zhe Yan et al., "LabanLab: An Interactive Choreographical System with
  Labanotation-Motion Preview", Eurographics Short Papers 2025.
- LaMoGen, 2026, which uses Labanotation as a symbolic bridge for language to
  motion generation and introduces symbolic evaluation.

What these papers collectively mean:

- Labanotation is already accepted as a useful symbolic interface.
- The nearby literature is moving from pure transcription toward symbolic
  editing, manipulation, and generation.
- However, this track remains body-motion oriented rather than finger-articulation
  oriented.

Implication for us:

- "Notation is useful" is not novelty anymore.
- "Hand articulation needs a sequence-native symbolic factorization distinct
  from body-motion notation pipelines" remains defensible.

## 3. Sign-language notation is nearby, but it is not the right target identity

Anchor systems:

- HamNoSys is a phonetic sign-language transcription system organized around
  handshape, orientation, location, movement, and non-manual features.
- SignWriting is a writing system for sign languages with symbols for
  handshape, movement, contact, and facial expression.

Nearby model trend:

- Recent sign-language work increasingly learns contextual and discrete
  representations rather than relying only on hand-written notation.
- SHuBERT (ACL 2025) pushes contextual self-supervised sign representations.
- A Data-Driven Representation for Sign Language Production (2024) learns a
  codebook of motion units from continuous 3D pose.
- Signs as Tokens (ICCV 2025) discretizes continuous signs into body-part token
  sequences for multilingual sign generation.
- M3T (2026) extends discrete tokenization to multi-modal sign streams,
  including non-manual channels.

Implication:

- If we frame temporal HL as "sign-like linguistic decomposition", reviewers can
  redirect the comparison toward sign-language notation and production.
- The safer identity is generic hand articulation, not linguistic sign
  structure.
- Our differentiation should stay on:
  - generic non-semantic hand motion
  - explicit anatomical factorization
  - human-readable symbolic temporal structure

## 4. Discrete and tokenized motion representation is now crowded

Representative neighboring works:

- Signs as Tokens, ICCV 2025.
- M3T, 2026.
- HOIGPT, CVPR 2025, uses a hand-object decomposed VQ-VAE tokenizer and a
  motion-aware language model for long-sequence HOI.
- TokenHand, CVPR 2026, uses discrete token representation for efficient hand
  mesh reconstruction.
- CoherentHand, CVPR 2026 Findings, explicitly criticizes generic VQ motion
  representations as too coarse for detailed temporal hand interactions.

Implication:

- "We discretize motion" is not enough.
- "We tokenize hand motion" is not enough.
- Reviewers have already seen learned tokenizers, codebooks, and motion tokens
  across sign, HOI, and hand mesh settings.

The only viable novelty direction here is not generic discreteness, but:

- interpretable symbolic structure
- anatomical decomposition
- sequence-native temporal channels
- controllability / editability / analysis advantages that opaque tokens do not
  give

## 5. Temporal hand modeling itself is also crowded

Representative works:

- Hierarchical Temporal Transformer for 3D Hand Pose Estimation and Action
  Recognition, CVPR 2023.
- HandFormer / On the Utility of 3D Hand Poses for Action Recognition, ECCV
  2024.
- Dyn-HaMR, CVPR 2025, recovers 4D interacting hand motion from monocular video
  with dynamic cameras.
- UniHand, ICLR 2026, unifies diverse controlled 4D hand motion estimation and
  generation under one diffusion framework.

Implication:

- "We model temporal hand motion better" is too broad and crowded.
- "We add a stronger temporal model" is also weak unless the representation
  itself exposes something these continuous models do not.
- Our advantage has to be symbolic and structural, not just temporal predictive
  power.

## 6. Strategic position that still looks open

After checking the current literature, the clearest open space is:

- a sequence-native symbolic hand representation
- anatomically factored rather than monolithic codebook tokens
- explicitly temporal at the representation level rather than only in the
  encoder
- human-readable and editable
- useful for retrieval, low-data transfer, and robot-control style interfaces

This is substantially stronger than:

- HL + temporal encoder
- learned motion tokenization without interpretability
- sign-language-style phonological decomposition
- generic 4D hand trajectory modeling

## 7. What this means for our experiment agenda

The literature scan aligns with the current local evidence:

- main contribution should be a representation upgrade, not a recognition
  upgrade
- state + transition is currently the cleanest temporal symbolic mainline
- duration and generic coordination features are weak unless reformulated
- intrinsic and low-data evidence matter because they are precisely where
  interpretable symbolic structure can outperform opaque learned tokens

Therefore, the strongest next-stage evaluation story should test whether the
representation:

- separates nearby motion classes better
- remains stable under lower-data training
- supports retrieval / indexing / editing better than frame-only HL
- retains hand-part interpretability that learned-token baselines lose
