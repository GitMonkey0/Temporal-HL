# Deep Related-Work Refresh for Temporal Hand Representation

Date: 2026-06-09

This is a research planning memo.
It is not paper text.

## Goal

Refresh the related-work landscape around upgrading Hand Labanotation (HL)
toward a stronger temporal representation, and decide which literature families
are:

- direct parents
- real novelty threats
- adjacent but not identity-defining
- useful foils for experiments

## Bottom line

After checking the 2023-2026 literature, the project should not be framed as:

- better HL recognition
- old HL plus a temporal encoder
- generic discrete tokenization of hand motion
- generic temporal hand modeling

The defensible space is narrower and clearer:

- sequence-native symbolic hand representation
- explicit temporal channels at the representation level
- anatomy-aware grouped factorization
- human-readable local editing and repair
- interacting-hand transition structure and controllable search

## 1. Direct parent work

### 1.1 Hand Labanotation is the immediate parent

The ACM MM 2024 Hand Labanotation paper already establishes:

- a 26-symbol hand-space inventory
- a large HLD dataset with more than 4 million annotated images
- automatic image-to-frame HL translation via MHLFormer

Implication:

- a follow-up centered on stronger frame-level HL translation is weak
- a follow-up centered on "temporal encoder on top of old labels" is exposed to
  incremental criticism
- a strong continuation must upgrade the representation itself rather than only
  the predictor

### 1.2 Data ancestry matters but is not the novelty

The original HL dataset is built from InterHand2.6M and FreiHAND. These remain
important because they define what kind of geometric supervision and coverage
the project inherits.

InterHand2.6M contributes:

- large-scale real captured single-hand and interacting-hand RGB data
- explicit emphasis on interacting hands rather than only isolated hands

FreiHAND contributes:

- a standard RGB hand pose and shape benchmark
- a real-image route to single-hand articulation coverage

Implication:

- reusing these data sources is acceptable
- but dataset reuse cannot carry the new submission
- the new value has to come from sequence structure, symbolic design, and
  downstream control/editability evidence

## 2. Temporal hand modeling is already crowded

This is one of the main novelty hazards.

### 2.1 HTT / hierarchical temporal modeling line

The CVPR 2023 Hierarchical Temporal Transformer line already treats temporal
information as central for:

- robust 3D hand pose estimation under self-occlusion
- action recognition over longer windows

Key lesson:

- temporal context helping hand understanding is not new
- "neighboring frames reduce ambiguity" is not new
- "joint pose plus temporal model helps actions" is not new

### 2.2 HandFormer / pose-driven action recognition line

ECCV 2024 HandFormer pushes 3D hand poses as a compact motion modality for
action recognition. It explicitly uses:

- dense high-temporal-resolution hand poses
- sparse RGB for scene semantics
- temporally factorized short-term joint trajectories

Key lesson:

- fine-grained temporal hand motion modeling is already a defended idea
- factorized temporal hand representation exists on the continuous side
- "we model hand motion efficiently over time" is not enough

### 2.3 Sequential pose recovery / 4D motion recovery line

Recent 2025-2026 work further crowds the temporal space:

- ICCV 2025 prior-aware dynamic temporal modeling for sequential 3D hand pose
  estimation
- CVPR 2025 Dyn-HaMR for 4D interacting hand motion recovery from a dynamic
  camera
- ICLR 2026 UniHand for diverse controlled 4D hand motion modeling

Key lesson:

- the field now expects explicit handling of occlusion, temporal continuity,
  and interaction dynamics
- a paper that sells only "better temporal hand modeling" will collide with a
  mature continuous-motion literature

Strategic implication:

- our contribution has to be representational and controllable, not merely
  temporally predictive

## 3. Discrete hand or sign tokens are now a major novelty threat

This is the second major novelty hazard.

### 3.1 Sign representation learning is now heavily tokenized

Recent sign-language representation work is increasingly discrete or clustered:

- ACL 2025 SHuBERT learns contextual sign representations via multi-stream
  cluster prediction over hand, face, and body pose streams
- ICCV 2025 Signs as Tokens discretizes continuous signs into body-part token
  sequences for multilingual sign generation

Key lesson:

- "we discretize motion" is no longer novel
- "we tokenize different body parts" is also no longer novel
- learned discrete sequence modeling is now normal in adjacent areas

### 3.2 Hand-specific discrete representations also appeared

CVPR 2026 TokenHand represents a 3D hand model using discrete tokens over
sub-structures for efficient single-view hand mesh reconstruction.

Key lesson:

- even within hand-specific vision, discrete sub-structure coding is now an
  accepted design pattern
- we should not frame Temporal HL as merely a hand tokenizer

Strategic implication:

The only version of "discrete" that still looks strong is:

- semantically explicit symbols rather than learned opaque codes
- anatomically grouped channels rather than monolithic token IDs
- local editability and interpretable repair
- transition-aware symbolic control that tokens do not naturally expose

## 4. Sign-language notation is adjacent, but it is the wrong identity

This line is important because reviewers may try to collapse Temporal HL into
sign-language notation rather than generic hand-motion representation.

### 4.1 Established notation systems

HamNoSys is a phonetic sign-language transcription system organized around
components such as handshape, orientation, location, movement, and non-manual
features.

SignWriting is a writing system for signed languages using visual symbols for
handshapes, movements, and facial expressions.

### 4.2 Model-based use of notation

CVPR 2023 Ham2Pose already turns HamNoSys strings into signed pose sequences.

Key lesson:

- notation-to-motion is not unique to our setting
- symbolic motion descriptions are already used in an operational generation
  loop

But the boundary is still real:

- sign notations are fundamentally organized around linguistic signs
- they are not designed as a generic symbolic interface for arbitrary
  non-semantic hand articulation
- they often entangle hand motion with language-specific structure and
  non-manual cues

Strategic implication:

The project should remain:

- anatomy-grounded
- non-semantic
- hand-motion centric
- useful for control, repair, retrieval, and editing even outside sign
  language

We should not drift into a phonological or linguistic framing.

## 5. Body-motion Labanotation remains relevant, but it does not solve hand articulation

Automatic body-motion Labanotation generation continues to progress:

- pose-estimation-based folk dance transcription in 2023
- LabanFormer in 2023
- hybrid transformer-LSTM generation work in 2026

Key lesson:

- symbolic motion notation is already validated as a useful computational
  interface
- recent work is moving from transcription toward stronger spatiotemporal
  models

But the gap remains:

- these systems are body-motion oriented
- they do not treat finger articulation and interacting-hand structure as
  first-class symbolic objects
- they do not naturally give a hand-specific local editing interface

Strategic implication:

- "notation is useful" is not our novelty
- "hand-specific temporal symbolic structure is still missing" remains
  defensible

## 6. 2026 pressure is shifting toward dexterous bimanual interaction

The 2026 frontier pushes strongly toward hand-specific and bimanual motion.

Representative pressure:

- CVPR 2026 HandX scales bimanual motion and interaction generation and argues
  existing resources miss detailed finger dynamics and interaction structure
- CVPR 2026 Text-Driven 3D Hand Motion Generation from Sign Language Data uses
  hand motion plus textual descriptions and motion-script cues
- ICLR 2026 UniHand unifies diverse controlled 4D hand motion modeling under a
  common generative framework

Key lesson:

- interacting-hand temporal structure is now clearly a central frontier topic
- if our representation leaves interaction timing implicit, it will look weak
- the hardest proving ground is no longer isolated hand motion

Strategic implication:

- interacting-hand temporal edits and compact search are exactly the right hard
  benchmark family
- right-hand interaction hard slices are not a distraction; they are where the
  project can become publication-worthy

## 7. What the literature says our claims cannot be

Unsafe main claims:

- "Temporal HL is better because the same classifier gets better sequence
  accuracy."
- "We model hand motion over time."
- "We discretize hand motion."
- "We use symbols, so the representation is interpretable."
- "Notation helps editing."

Each of these already has strong neighboring literature or is too generic.

## 8. What the literature says our claims can be

Safer, stronger claims:

- the representation itself carries explicit temporal semantics rather than
  outsourcing time to the encoder
- anatomy-aware grouped symbolic structure yields cleaner local control than
  flat or opaque alternatives
- symbolic transition channels enable transition-conditioned editing and compact
  search over interacting-hand motions
- human-readable factorization enables zero-harm repair and controllable
  local-to-global manipulation

These claims align with the current local evidence much better than generic
classification or retrieval claims.

## 9. Concrete reviewer-collision map

### Collision A: old HL plus temporal model

Closest literature:

- Hand Labanotation 2024
- HTT 2023
- HandFormer 2024
- sequential 3D hand pose work in 2025

What reviewers will say:

- the labels are old
- only the encoder changed
- this is standard temporal modeling

Required answer:

- explicit new temporal symbolic channels
- matched old-HL vs temporal-HL comparison
- downstream tasks that consume symbolic transitions directly

### Collision B: just another motion tokenizer

Closest literature:

- SHuBERT 2025
- Signs as Tokens 2025
- TokenHand 2026

What reviewers will say:

- this is only a hand-specific discrete code
- learned tokens already solve sequence modeling

Required answer:

- local editability
- interpretable repair
- channel semantics
- compact search and transplant tasks where opacity is a weakness

### Collision C: generic temporal hand modeling

Closest literature:

- HTT 2023
- HandFormer 2024
- Dyn-HaMR 2025
- UniHand 2026

What reviewers will say:

- continuous pipelines already model time and interaction

Required answer:

- show what those pipelines do not naturally expose:
  - explicit symbolic transitions
  - localized manual edits
  - family-level correction
  - interaction-aware compact search with readable intermediate states

### Collision D: sign-language notation identity collapse

Closest literature:

- HamNoSys
- SignWriting
- Ham2Pose
- sign tokenization papers

What reviewers will say:

- this is notation for signing, not a generic hand representation

Required answer:

- stay non-semantic
- stay anatomy-grounded
- use generic hand-motion tasks rather than sign-semantic tasks as the main
  proof

## 10. Practical experiment guidance implied by this survey

The literature does not demand more raw sequence classification.

It does demand evidence that the representation enables something not already
covered by:

- frame-wise HL translation
- continuous temporal hand pipelines
- learned discrete tokenizers

That means the most valuable experiment families are:

- transition-conditioned editing
- local symbolic repair
- compact search on hard interacting-hand slices
- interaction-aware counterfactual transfer
- reconstruction or control interfaces where readability matters

The least valuable experiment families are:

- one more classifier on top of the same labels
- one more temporal encoder sweep
- generic retrieval without edit or control interpretation

## 11. Final positioning recommendation

As of 2026-06-09, the project should be positioned internally as:

"A sequence-native symbolic hand representation with explicit temporal channels,
grouped anatomy, and controllable interacting-hand edits."

This is stronger than:

- temporal HL as a better classifier input
- temporal HL as a hand tokenizer
- temporal HL as sign-like motion phonology

And it is the position most consistent with both:

- the external literature
- the current local evidence stack
