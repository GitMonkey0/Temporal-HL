# Related Work Matrix for Temporal Hand Labanotation

Date: 2026-06-09

This file is a research memo for project planning and experiment design.
It is not paper text.

## Goal

Clarify which existing work is:

- a direct baseline we must cite and beat
- a dangerous neighboring area that can collapse our novelty
- a supporting area that helps motivate the project but should not define it

Target project direction:

- upgrading frame-wise Hand Labanotation into a sequence-native hand
  representation with explicit temporal structure

## Executive takeaways

1. The direct continuation of HL is crowded if framed as recognition.
2. The most dangerous neighboring area is discrete motion tokenization.
3. The second most dangerous neighboring area is generic temporal hand modeling.
4. The safest novelty wedge remains:
   - sequence-native symbolic hand representation
   - explicit anatomical factorization
   - temporal semantics at the representation level
   - human readability / editability / controllability
5. The strongest evaluation pressure from the literature is not just accuracy:
   - low-data transfer
   - retrieval / indexing
   - hard-slice robustness
   - interpretability relative to opaque learned tokens

## A. Direct baselines

### A1. Hand Labanotation

- Ling Li et al., "Translating Motion to Notation: Hand Labanotation for
  Intuitive and Comprehensive Hand Movement Documentation", ACM MM 2024.
- Why it matters:
  - this is the direct parent work
  - it already owns frame-wise hand notation, HLD, and image-to-HL translation
- Risk:
  - if we only improve the translator or add a temporal encoder, the work will
    look incremental
- Required response:
  - the label space itself must change
  - experiments must show representation gain, not only model gain

### A2. Automatic body-motion Labanotation

- Xingquan Cai et al., "Automatic generation of Labanotation based on human
  pose estimation in folk dance videos", Neural Computing and Applications,
  2023.
- Junkun Jiang et al., "Motion Part-Level Interpolation and Manipulation over
  Automatic Symbolic Labanotation Annotation", IJCNN 2024.
- Zhe Yan et al., "LabanLab: An Interactive Choreographical System with
  Labanotation-Motion Preview", Eurographics Short Papers 2025.
- Junkun Jiang et al., "LaMoGen: Language to Motion Generation Through
  LLM-Guided Symbolic Inference", arXiv 2026.
- Why they matter:
  - prove notation is already accepted as an interface for motion processing
  - show the community is moving from transcription to manipulation and
    generation
- Risk:
  - "symbolic motion interface" alone is not novelty anymore
- Required response:
  - hand articulation must be argued as structurally different from body-level
    notation pipelines

## B. Adjacent symbolic systems that are not our identity

### B1. HamNoSys

- Hamburg Sign Language Notation System, University of Hamburg.
- Why it matters:
  - reviewer fallback when they hear "symbolic hand notation"
  - it is a mature phonetic transcription system for sign languages
- Risk:
  - if we drift toward linguistic decomposition, the work gets redirected into
    sign-language notation territory
- Required response:
  - stay anchored to generic non-semantic hand articulation
  - do not define the project around language units

### B2. SignWriting

- Sutton SignWriting and ISWA resources.
- Why it matters:
  - demonstrates that symbolic systems covering handshape, movement, and facial
    expression already exist
- Risk:
  - "we made a readable symbol inventory" is not enough
- Required response:
  - emphasize anatomy-aware temporal coding for arbitrary hand motion rather
    than a writing system for signed languages

## C. Most dangerous novelty neighbors: discrete tokenization

### C1. Sign-language tokenization

- Artem Abzaliev and Rada Mihalcea, "Unsupervised Discrete Representations of
  American Sign Language", EMNLP 2024.
- Harry Walsh et al., "A Data-Driven Representation for Sign Language
  Production", arXiv 2024.
- Ronglai Zuo et al., "Signs as Tokens: A Retrieval-Enhanced Multilingual Sign
  Language Generator", ICCV 2025.
- Shester Gueuwou et al., "SHuBERT: Self-Supervised Sign Language
  Representation Learning via Multi-Stream Cluster Prediction", ACL 2025.
- Alexandre Symeonidis-Herzig et al., "M3T: Discrete Multi-Modal Motion Tokens
  for Sign Language Production", arXiv 2026.
- Why they matter:
  - they already legitimize continuous sign motion to discrete token streams
  - several of them are sequence-native and part-aware
- Risk:
  - reviewers may reduce temporal HL to "another tokenizer"
- Required response:
  - compare against the learned-token framing directly in discussion
  - show what explicit symbols buy that learned codes do not:
    interpretability, editability, controllability, stable low-data transfer,
    family-level error analysis

### C2. Hand / HOI tokenization

- Huang et al., "HOIGPT: Learning Long-Sequence Hand-Object Interaction with
  Language Models", CVPR 2025.
- Xinguo He et al., "TokenHand: Discrete Token Representation for Efficient
  Hand Mesh Reconstruction", CVPR 2026.
- Bikram Boote et al., "CoherentHand: Temporally Consistent 3D Hand Trajectory
  Synthesis with Semantic Motion Priors", CVPR Findings 2026.
- Why they matter:
  - tokenization has already entered hand-specific and HOI-specific pipelines
  - CoherentHand explicitly argues that generic discrete motion prediction can
    be temporally noisy or too coarse
- Risk:
  - if our story is only compression, tokenization, or sequence modeling, it
    will be absorbed by this literature
- Required response:
  - anchor the contribution on human-readable symbolic structure and temporal
    semantics rather than generic discrete coding

## D. Dangerous but secondary neighbors: temporal hand modeling

### D1. Temporal pose / action modeling

- Yilin Wen et al., "Hierarchical Temporal Transformer for 3D Hand Pose
  Estimation and Action Recognition From Egocentric RGB Videos", CVPR 2023.
- Aditya Prakash et al., "3D Hand Pose Estimation in Everyday Egocentric
  Images", ECCV 2024.
- Shamil Mamedov et al., "On the Utility of 3D Hand Poses for Action
  Recognition", ECCV 2024. The model is HandFormer.
- Zhengdi Yu et al., "Dyn-HaMR: Recovering 4D Interacting Hand Motion from a
  Dynamic Camera", CVPR 2025.
- Zhihao Sun et al., "UniHand: A Unified Model for Diverse Controlled 4D Hand
  Motion Modeling", ICLR 2026.
- Why they matter:
  - temporal hand motion is already an active and competitive area
  - sequence modeling alone is therefore weak novelty
- Risk:
  - "we model hand motion better over time" is too generic
- Required response:
  - representation-level temporal semantics must be central
  - use tasks where symbolic factorization matters more than raw trajectory
    fidelity alone

### D2. Sequential 3D hand estimation under sparse or non-visual inputs

- Wang et al., "MGDHand: Multi-Granularity Prior-to-Inertial Distillation
  Framework for Sequential 3D Hand Pose Estimation", CVPR 2026.
- Why it matters:
  - further evidence that sequence priors for hand motion are already crowded
- Required response:
  - avoid claiming novelty on temporal smoothness or sequential consistency
    alone

## E. Strategic contribution space that still looks open

The literature still leaves room for the following combination:

- frame-wise HL upgraded into a sequence-native symbol system
- explicit state + transition + event / persistence structure
- anatomy-aware factorization rather than a monolithic codebook
- direct human readability
- support for editing / retrieval / low-data transfer / control smoothing

This space is stronger than:

- better image-to-HL translation
- HL plus temporal transformer
- generic VQ tokenizer for hand motion
- generic 4D hand modeling

## F. What this literature implies for our experiments

### F1. Must-have evidence

- old HL vs temporalized HL under matched downstream protocols
- temporalized HL vs old HL plus stronger temporal encoder
- low-data comparisons
- hard-slice analysis:
  - occlusion
  - right vs left hand
  - wrist ROM
  - finger occlusion / articulation families
- representation diagnostics:
  - retrieval
  - confusion-family repair
  - stability under threshold changes

### F2. High-value but probably necessary soon

- at least one comparison against a learned-token baseline or a strong proxy
  that stands in for learned tokenization
- a control-facing or editability-facing experiment showing why symbolic
  temporal channels are useful beyond recognition

### F3. Claims we should avoid

- "first discrete hand motion representation"
- "first temporal hand motion model"
- "first symbolic motion interface"
- "first motion tokenizer for hands"

These are all exposed by existing literature.

## G. Reviewer attack map

### Attack 1: "This is just HL plus temporal encoder"

Prevention:

- new representation channels must be explicit
- downstream tasks must depend on those channels

### Attack 2: "This is just another motion tokenizer"

Prevention:

- emphasize human readability and anatomy-aware semantics
- include analyses that require explicit symbols

### Attack 3: "Why not compare to sign-language notation?"

Prevention:

- keep the target identity as generic hand articulation, not signed language
- explain that sign systems are semantic and linguistic, while our target is
  action encoding

### Attack 4: "Why not just use continuous 4D hand models?"

Prevention:

- focus on settings where explicit symbolic abstraction is helpful:
  low data, retrieval, controllable editing, control smoothing, family-level
  analysis

## H. Bottom line for project planning

The literature supports a strong project only if we keep the center of gravity
on representation design rather than recognizer design.

The current best thesis is:

- not frame-wise HL
- not generic tokenization
- not generic temporal modeling
- but sequence-native, anatomy-aware, interpretable symbolic hand
  representation with explicit temporal structure
