# Related Work Survey for Temporal HL

Date: 2026-06-09

This file is a research memo, not paper text.

## Objective

Map the external literature around a temporal upgrade of Hand Labanotation
(HL), with emphasis on:

- symbolic / notation-based hand and movement representation
- temporal hand pose / hand reconstruction
- sequence modeling for hand action and hand interaction
- discrete tokenization of motion

The goal is to decide what counts as a real representation contribution rather
than a generic sequence-modeling increment.

## Local project status that matters for this survey

Current local work under `/opt/tiger/hand` has already moved beyond frame-wise
HL:

- the roadmap explicitly rejects "just add a temporal transformer" as too weak
- current exported labels already include state, transition, persistence,
  segment-duration, and interaction-motion style factors
- the strongest local evidence currently favors:
  - sequence-native symbolic grouping
  - anatomy-aware factorization
  - stable family-level repair
  - local editability / controllability

This matters because the literature review should be used to sharpen the
representation claim, not to drift back toward plain temporal prediction.

## Literature map

### 1. Classical notation systems: strong inspiration, weak direct fit

Relevant systems:

- Labanotation
- HamNoSys
- SignWriting / Sutton Movement Writing

What they contribute:

- Labanotation is a mature movement notation system that explicitly binds
  symbols to direction, body parts, level, and timing.
- HamNoSys is a phonetic transcription system for signed languages.
- SignWriting is a writing system for signed languages with symbol categories
  for handshape and movement.

What they do **not** solve for us:

- They are built for whole-body dance or signed-language transcription, not for
  generic hand-motion documentation as a machine-actionable intermediate
  representation.
- HamNoSys / SignWriting are linguistically grounded; they are useful evidence
  that symbolic hand-related systems exist, but they are not optimized for
  exhaustive, non-semantic hand kinematics.
- These systems give us precedent for discrete symbols and timing-aware
  notation, but not a ready-made sequence-native hand representation for modern
  CV / robotics pipelines.

Takeaway:

- Use them as legitimacy for symbolic movement representation.
- Do not frame temporal HL as "another sign-language notation".
- The gap remains: a hand-specific, sequence-native, anatomically grounded
  representation that is both editable and machine-usable.

### 2. Hand pose / hand reconstruction with temporal cues: strong baseline class, but not representation work

Recent lines:

- HaMeR / modern transformer-based hand reconstruction
- temporal low-resolution hand reconstruction
- interacting hand-object video reconstruction

Common pattern:

- time is used as evidence aggregation for better geometry recovery
- output stays in coordinates / MANO / mesh space
- gains are framed as robustness, reconstruction accuracy, or handling
  occlusion / low resolution

Why this matters:

- These works are strong baselines against any claim that "temporal context for
  hands is novel"
- They do **not** compete directly on symbolic representation or editability
- They suggest the right standard for rigor: temporal HL should be evaluated on
  hard sequence settings such as low resolution, occlusion, or interacting
  hands, not just clean static poses

Takeaway:

- Temporal modeling alone is not novel enough.
- If we only predict HL per frame with a better sequence model, reviewers can
  place us into this bucket and dismiss the work as another temporal CV model.

### 3. Hand action / hand interaction sequence modeling: closest neighboring task

Recent lines:

- HandFormer uses high-rate 3D hand poses plus sparse RGB for action modeling
- hierarchical temporal transformers jointly model pose and action
- interacting hand / object recognition methods increasingly emphasize temporal
  context and contact dynamics

What this line shows:

- 3D hand pose is already accepted as a compact action-relevant modality
- sequence structure matters for disambiguating hand-object interaction
- multimodal context remains important because pose alone is often incomplete

Why this matters for temporal HL:

- This is the most natural neighboring community for a sequence-native hand
  representation claim
- A strong temporal-HL paper should not stop at reconstruction or recognition
  accuracy; it should show that symbolic temporal structure supports sequence
  comparison, repair, editing, and possibly control

Takeaway:

- Temporal HL should position itself as a structured alternative to raw pose
  sequences for hand-action understanding and manipulation-oriented reasoning
- The likely comparison axis is not only "accuracy", but also compactness,
  repairability, edit locality, and transfer to control

### 4. Sign-language computational modeling: useful cautionary evidence

Recent lines:

- explicit handshape modeling is becoming more important in sign-language
  processing
- recent handshape work separates static configuration from temporal dynamics
- skeleton-based CSLR work keeps emphasizing hand-centric graph structure and
  temporal generalization challenges
- HamNoSys is still used as a phonological representation for analysis and
  motion editing

What this line contributes:

- the community increasingly recognizes that handshape should not be treated as
  a by-product of whole-body sequence models
- there is precedent for separating static hand configuration from temporal
  evolution
- symbolic linguistic representations can support downstream editing

Why this is still not our target:

- sign-language pipelines optimize for lexical / linguistic meaning
- they assume a phonological inventory tied to signed language
- non-manual channels are integral there, while our target is a general hand
  movement representation

Takeaway:

- Borrow the structural lesson: separate hand state from temporal change.
- Avoid inheriting the semantic burden of sign-language notation.

### 5. Discrete motion tokenization: biggest competitive threat

Relevant lines:

- tokenized pose representations such as TokenHMR
- motion-token papers such as Moto
- hand-specific movement tokenization such as VQ-MyoPose

What this line shows:

- the field is increasingly comfortable with learned discrete codes for motion
- tokenization is already being sold as a bridge between geometry and sequence
  modeling
- hand movement can also be discretized into learned codebooks

Why this is a threat:

- if temporal HL is framed only as "discretize hand motion into symbols", it
  risks being read as a manual codebook alternative to learned tokens
- learned token methods can be stronger on pure discriminability or compression

Why this is also an opportunity:

- learned tokens are usually opaque
- they are weak on user-editability, semantics-preserving local control, and
  anatomy-explicit factorization unless extra machinery is added
- this matches the current local evidence, where symbolic structure is already
  strongest on repair and local edits rather than raw classification

Takeaway:

- Do not center the project on "symbolic beats tokenized on accuracy".
- Center it on interpretable temporal factorization, stable correction, and
  controllable local editing.

## What the literature implies for Temporal HL

### A. The representation upgrade must be explicit

A defensible temporal HL should encode time in the representation itself, not
just in the model. The literature already has many temporal encoders for hand
vision tasks. To avoid looking incremental, the symbolic inventory should make
at least one of the following first-class:

- transition type
- persistence / duration
- segment boundaries
- coordination across fingers or across hands
- interaction dynamics

### B. The state / change split is well justified

The sign-language and action-recognition literature both support a split
between static hand configuration and temporal dynamics. This aligns with the
current local direction of:

- per-frame state
- adjacent-frame transition
- persistence / duration
- interaction motion

This looks more defensible than a monolithic sequence token.

### C. Pure recognition is the wrong main battleground

The surrounding literature is strong on:

- recognition accuracy
- reconstruction accuracy
- motion modeling with learned tokens

Temporal HL is more likely to stand out on:

- interpretable sequence structure
- zero-harm family repair
- predictable local edits
- geometry-aware locality under symbolic intervention
- hand-to-robot / manipulation transfer

### D. Interacting-hand and manipulation settings are strategically important

The recent hand literature keeps moving toward:

- interacting hands
- hand-object interaction
- egocentric and in-the-wild video
- transfer to dexterous manipulation

A temporal-HL story that stays only on isolated single-hand motion will look
too small. The strongest external fit is likely:

- bimanual or interacting-hand motion
- interaction change over time
- manipulation-oriented symbolic control

## Candidate novelty wedges after this survey

Ordered from strongest to weakest.

### 1. Sequence-native symbolic hand representation

Definition:

- temporal HL is a symbolic sequence language whose primitives are hand state,
  transition, persistence, and interaction events

Why it survives review pressure:

- distinct from frame-wise pose labels
- distinct from generic temporal transformers
- distinct from opaque learned tokens

### 2. Anatomy-aware editable temporal factorization

Definition:

- the representation separates stable hand state, local change, and
  cross-hand interaction into editable symbolic channels

Why it survives review pressure:

- aligns with hand physiology
- supports local intervention claims
- matches current local evidence better than accuracy-only framing

### 3. Symbolic repair and control as the main downstream proof

Definition:

- show that temporal symbolic structure supports stable error repair and local
  controllable edits without collateral drift

Why it survives review pressure:

- learned tokens and coordinate regressors do not get this "for free"
- this is where current local experiments are already strongest

## Claims that the survey suggests we should avoid

- "Temporal context for hand understanding is novel."
- "Discrete hand representation is novel."
- "A temporal transformer over HL is a representation contribution."
- "Sign-language notation already solves this problem."
- "Symbolic representation should win mainly on classification accuracy."

## Immediate experimental consequences

### Highest-priority directions

- strengthen the sequence-native nature of the representation itself
- push the control / repair / local-edit line harder
- keep geometry-aware verification for edit locality
- add more interacting-hand and manipulation-relevant evaluation slices

### Good comparison families

- frame-wise HL + temporal encoder
- raw pose / MANO sequence baselines
- learned token baselines
- sign-inspired factor baselines when feasible

### Lower-priority directions

- another round of generic temporal backbone tuning
- purely frame-level classification improvements
- cosmetic extensions that do not alter the representation

## Bottom line

After this survey, the most defensible direction is not:

- "HL + stronger sequence model"

It is:

- "a sequence-native symbolic hand representation with explicit temporal
  structure that is interpretable, editable, stable under repair, and useful as
  an intermediate language for interaction reasoning and control"
