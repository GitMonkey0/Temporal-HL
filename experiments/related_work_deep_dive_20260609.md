# Related Work Deep Dive for Temporal Hand Labanotation

Date: 2026-06-09

This file is a research memo for project planning. It is not paper text.

## Scope

Question:

- If we upgrade Hand Labanotation (HL) from a frame-wise notation into a
  sequence-native representation, what related work is closest, what novelty
  is already crowded, and what contribution space still looks strong enough
  for AAAI 2027?

This memo only records positioning and experiment implications. It does not
draft claims for a paper.

## Executive conclusion

The literature does not support a weak story like:

- better image-to-HL recognition
- HL plus temporal encoder
- hand-motion discretization by itself

Those spaces are already surrounded by:

- HL itself as a frame-wise symbolic hand notation
- automatic Labanotation systems for full-body motion
- sign-language notation and sign tokenization
- hand and HOI discrete token pipelines
- continuous 4D hand-motion modeling

The clearest open space remains:

- sequence-native symbolic hand representation
- explicit anatomical factorization
- temporal semantics inside the representation, not only in the encoder
- human readability and editability
- evidence on retrieval / low-data transfer / control-facing stability

## 1. Direct baseline: Hand Labanotation already owns frame-wise symbolic hand notation

- The ACM MM 2024 Hand Labanotation paper already defines HL as a symbolic hand
  notation, introduces HLD with 4M annotated images, and proposes MHLFormer for
  image-to-frame HL translation.
- The published identity of that work is notation inventory plus multi-view
  translation, not sequence-native symbolic reasoning.

Implication:

- A continuation centered on recognition quality will look incremental.
- A continuation centered on temporal encoders over the same per-frame label
  space will also look incremental.
- The representation itself has to change.

## 2. Automatic Labanotation is active, but mostly body-centric

Representative papers:

- Cai et al., Neural Computing and Applications 2023:
  automatic Labanotation generation from folk-dance videos using pose
  estimation and keyframe-style processing.
- Jiang et al., IJCNN 2024:
  motion part-level interpolation and manipulation over automatic symbolic
  Labanotation annotation.
- Yan et al., Eurographics 2025 short paper:
  LabanLab, an interactive choreography system with Labanotation-motion preview.
- Jiang et al., CVPR 2026:
  LaMoGen, which introduces LabanLite and uses symbolic reasoning as a bridge
  from language to motion generation.

Common pattern:

- Labanotation is accepted as a useful symbolic interface.
- The literature is shifting from pure transcription toward symbolic editing,
  manipulation, and generation.
- The representation target is still body motion, dance, or text-to-motion,
  not fine-grained finger articulation as a first-class symbolic object.

Implication:

- "Notation is useful" is no longer novelty.
- "Hand articulation requires a different notation granularity and temporal
  factorization than body-motion Labanotation" remains viable.

## 3. Sign-language notation is adjacent, but should not become our identity

Anchor systems:

- HamNoSys is a phonetic transcription system for sign languages and is widely
  used in that role.
- SignWriting is a writing system for signed languages with symbols covering
  handshape, movement, and facial-expression related content.

Why they matter:

- They are the obvious reviewer fallback if we blur the line between generic
  hand articulation and linguistic sign representation.
- They prove that symbolic hand-related notation is not novel by itself.

Why they are still not the right center:

- Their target is language, not generic non-semantic hand articulation.
- They include linguistic structure, body location, and non-manual components
  that are essential in sign languages but not mandatory for generic hand
  motion documentation.

Positioning implication:

- Temporal HL should stay anchored to generic hand articulation.
- Avoid framing the work as phonological sign decomposition.
- Avoid centering semantic gesture recognition as the main identity.

## 4. Discrete motion tokenization is the most dangerous neighboring area

### 4.1 Sign-language tokenization

Representative works:

- Abzaliev and Mihalcea, EMNLP 2024:
  unsupervised discrete representations of ASL fingerspelling.
- Walsh et al., 2024:
  a data-driven representation for sign language production via VQ codebooks.
- Zuo et al., ICCV 2025:
  Signs as Tokens (SOKE), which discretizes continuous signs into body-part
  token sequences for multilingual sign generation.
- Gueuwou et al., ACL 2025:
  SHuBERT, a self-supervised sign representation model over clustered
  hand/face/body streams.
- Symeonidis-Herzig et al., 2026:
  M3T, discrete multi-modal motion tokens for sign language production with
  explicit non-manual channels.

Implication:

- "Continuous motion to discrete tokens" is already normal.
- Learned tokenizers and codebooks already cover body-part tokenization and
  sequence generation.
- If temporal HL is presented as another tokenization method, reviewers can
  collapse it into this crowded line immediately.

What remains differentiable:

- human-readable symbolic structure
- anatomically constrained factorization
- explicit temporal semantics such as transition, persistence, or coordination
- interpretability, editability, and controllability advantages

### 4.2 Hand and HOI tokenization

Representative works:

- Huang et al., CVPR 2025:
  HOIGPT with a hand-object decomposed VQ-VAE tokenizer and motion-aware LM.
- He et al., CVPR 2026:
  TokenHand for efficient hand mesh reconstruction via discrete token
  representation.
- Boote et al., CVPR 2026 Findings:
  CoherentHand, which explicitly argues that generic discrete motion generators
  are too coarse and proposes semantically enriched motion priors.

Implication:

- Reviewers in CV/ML have already seen discrete hand and HOI token pipelines.
- The token story is no longer enough even inside hand-specific literature.
- A strong symbolic hand project must emphasize interpretability and analysis
  value, not only compression or generation convenience.

## 5. Temporal hand modeling itself is crowded

Representative works:

- Wen et al., CVPR 2023:
  Hierarchical Temporal Transformer for 3D hand pose estimation and action
  recognition from egocentric RGB.
- Shamil et al., ECCV 2024:
  HandFormer, which temporally factorizes hand modeling by representing each
  joint with short-term trajectories and combining dense hand-pose streams with
  sparse RGB.
- Yu et al., CVPR 2025:
  Dyn-HaMR for 4D interacting hand motion recovery from monocular dynamic
  cameras.
- Sun et al., ICLR 2026:
  UniHand, a unified diffusion-style framework for diverse controlled 4D hand
  motion modeling.

Implication:

- "We model temporal hand motion better" is too generic.
- "We add a temporal transformer" is not defensible as a main contribution.
- The differentiator must be representation structure rather than temporal
  modeling power alone.

## 6. Strongest still-open novelty wedge

After this scan, the most defensible wedge is:

- a sequence-native extension of HL
- representation-level temporal channels, not only sequence encoders
- anatomy-aware symbol design instead of opaque learned codes
- utility on tasks where interpretability matters:
  retrieval, low-data transfer, symbolic editing, reconstruction constraints,
  or control smoothing

This wedge is stronger than:

- improving MHLFormer
- replacing MHLFormer with a better sequence model
- generic VQ tokenization of hand trajectories
- generic 4D hand-motion estimation

## 7. What experiments the literature is implicitly demanding

Because nearby work already covers tokenization and temporal modeling, a strong
temporal-HL submission will likely need evidence for all of the following:

- representation benefit beyond frame-wise HL
- benefit beyond "more temporal context" baselines
- interpretability benefit relative to learned tokenizers or opaque sequence
  encoders
- robustness under low-data or hard-slice settings
- hand-part or event-level analysis that learned latent tokens do not expose

Concrete experiment pressure:

- old HL vs temporal HL under matched downstream protocols
- temporal HL vs plain temporal encoder on old HL labels
- temporal HL vs strong continuous hand-motion baselines on at least one
  downstream task
- slice analysis on occlusion, interaction, wrist ROM, and finger articulation
- retrieval or matching tasks where symbolic event structure should help

## 8. Review risks to actively avoid

### Risk A: "This is just HL + temporal encoder"

How it happens:

- representation unchanged
- only architecture or training upgraded

What prevents it:

- transition / event / persistence / coordination as explicit symbols
- evaluation that depends on those symbols

### Risk B: "This is just another motion tokenizer"

How it happens:

- learned codes dominate the story
- human readability is not central

What prevents it:

- symbolic channels with direct anatomical semantics
- analysis tasks that use those semantics directly

### Risk C: "This belongs to sign-language notation instead"

How it happens:

- overuse of linguistic language
- semantic gesture framing becomes primary

What prevents it:

- generic non-semantic hand articulation tasks
- hand-motion documentation, retrieval, and control framing

### Risk D: "This is useful only as a toy symbolic layer"

How it happens:

- only visualization demos
- no strong downstream or hard-case evidence

What prevents it:

- strong low-data transfer
- retrieval hard-case gains
- control-facing reconstruction or stability evidence

## 9. Immediate project guidance

Based on the local experimental record and this literature scan:

- Do not spend the main contribution budget on better frame-wise HL
  translation.
- Do not center the story on a new encoder alone.
- Keep sequence-native symbolic structure as the core.
- Keep sign-language work in related work, not as the target identity.
- Treat discrete-token literature as the main novelty threat.
- Build evidence where symbolic structure has an unfair advantage:
  interpretability, low-data, retrieval, and hard-case anatomy-aware analysis.

## 10. Shortlist of anchor references to cite later

Direct baseline:

- Li et al., ACM MM 2024, Hand Labanotation

Automatic Labanotation / symbolic motion interface:

- Cai et al., Neural Computing and Applications 2023
- Jiang et al., IJCNN 2024
- Yan et al., Eurographics 2025
- Jiang et al., CVPR 2026, LaMoGen

Sign notation / sign representation:

- HamNoSys
- SignWriting
- Abzaliev and Mihalcea, EMNLP 2024
- Walsh et al., 2024
- Gueuwou et al., ACL 2025
- Zuo et al., ICCV 2025
- Symeonidis-Herzig et al., 2026

Temporal hand / HOI modeling:

- Wen et al., CVPR 2023
- Shamil et al., ECCV 2024
- Yu et al., CVPR 2025
- Huang et al., CVPR 2025
- He et al., CVPR 2026
- Boote et al., CVPR 2026 Findings
- Sun et al., ICLR 2026
