# Related Work Scout for Temporal Hand Labanotation

Date: 2026-06-09

This file is a research memo for experiment planning. It is not paper text.

## Scope

Target question:

- If HL is upgraded from frame-wise hand symbols to a sequence-native
  representation, what existing work already covers nearby space, and what
  contribution space is still open enough for a strong AAAI 2027 submission?

This memo focuses on six lines:

1. Hand Labanotation itself
2. Human-motion notation / automatic Labanotation
3. Sign-language notation and symbolic editing
4. Discrete / tokenized motion representations
5. Temporal hand-pose and hand-motion modeling
6. Robot-control interfaces based on compact pose / motion codes

## 1. Hand Labanotation baseline

Primary reference:

- Ling Li et al., "Translating Motion to Notation: Hand Labanotation for
  Intuitive and Comprehensive Hand Movement Documentation", ACM MM 2024.
- Key released artifacts in the paper:
  HLD (>4M annotated images), a 26-symbol hand-space inventory, and MHLFormer
  for image-to-framewise-HL translation.

What matters for us:

- HL already defines a frame-wise symbolic inventory for hand regional vectors.
- The released story is mainly "image -> per-frame HL translation", with HLD and
  MHLFormer as the central artifacts.
- The strongest direct continuity path is not "better recognizer", but
  "stronger representation layer than frame-wise HL".

Implication:

- Any AAAI 2027 upgrade that only adds sequence modeling on top of the old label
  space will look incremental.
- The safe direction is to move temporal information into the representation
  itself: transition, duration, phase, coordination, or segment structure.

Concrete review risk:

- Reviewers can reasonably say "this is HL + temporal encoder" unless the new
  label space itself changes.
- Therefore, temporal HL must be framed as a representation upgrade, not an
  architecture refresh.

## 2. Human-motion notation and automatic Labanotation

Observed pattern from adjacent Labanotation literature:

- Existing automatic Labanotation work remains concentrated on full-body or
  dance settings.
- Common pipelines still revolve around pose extraction, motion segmentation,
  then symbol generation.
- Recent work improves automation, but the notation target is usually still a
  mostly frame/segment-wise symbolic transcription rather than a dedicated
  sequence-native representation.

What this means for temporal HL:

- We do not need to "beat the entire Labanotation literature" on full-body
  motion.
- We do need to show that hand motion is structurally different enough that
  simple reuse of body-motion Labanotation pipelines is insufficient.
- A key argument should be that fingers exhibit high-frequency local transitions,
  cross-finger coordination, and short stable-to-change cycles that body-level
  Labanotation systems were not designed to encode.

Concrete papers to anchor:

- Cai et al., "Automatic generation of Labanotation based on human pose
  estimation in folk dance videos", Neural Computing and Applications 2023.
  This is a pose-estimation -> keyframe extraction -> symbol generation style
  pipeline on folk dance video, which is useful as evidence that automatic
  Labanotation remains mostly full-body and pipeline-centric.
- Jiang et al., "Motion Part-Level Interpolation and Manipulation over
  Automatic Symbolic Labanotation Annotation", IJCNN 2024.
  This is important because it treats Labanotation as an editable symbolic
  interface for motion manipulation rather than just archival scoring.
- Yan et al., "LabanLab: An Interactive Choreographical System with
  Labanotation-Motion Preview", Eurographics 2025 short paper.
  This is relevant because it shows the community moving toward symbolic
  motion interfaces with interactive preview, but again in dance/full-body
  settings.
- Earlier sequential generation work such as DFGNN-CTC for continuous
  Labanotation generation is also relevant as evidence that long unsegmented
  body-motion transcription is studied, but not finger-level sequence-native
  articulation coding.

Strategic takeaway:

- These papers make it harder to sell "notation is useful" as a novelty.
- They also make it easier to sell "hand articulation needs a different
  notation granularity and temporal factorization than body choreography".

## 3. Sign-language notation and symbolic editing

Important nearby systems:

- HamNoSys remains the best-known symbolic notation system for sign language.
- SignWriting remains the broad symbolic writing system with body, hand, and
  movement coverage.

Concrete anchor facts:

- HamNoSys is explicitly a phonetic transcription system for sign languages and
  organizes notation around handshape, orientation, location, movement, and
  non-manual components.
- SignWriting is a writing system for sign languages, with symbols for
  handshape, movement, contact, and facial expressions, and is tied to
  language use rather than generic non-semantic hand articulation.

What nearby 2024-2025 work is doing:

- HamNoSys-based motion editing uses symbolic components to edit MoCap-driven
  sign motion rather than documenting arbitrary non-semantic hand motion.
- Recent sign-language work also studies symbolic or sub-unit-level
  representations, but usually in the service of linguistic modeling, sign
  production, or sign generation.

Key distinction to protect:

- HL is not a sign-language notation replacement.
- Temporal HL should avoid drifting toward "phonological sign representation",
  because that invites a direct comparison against sign-language notation
  systems where semantics, location, facial expression, and linguistic units are
  primary.
- The clean positioning is:
  sequence-native symbolic encoding of generic hand articulation, usable for
  documentation, retrieval, classification transfer, and robot-control
  interfaces.

Concrete review risk:

- If we overemphasize linguistic units, reviewers can redirect us to the sign
  language community and ask why HamNoSys / SignWriting / phonological sign
  decomposition is not the more natural framing.
- If we overemphasize semantic gesture classes, reviewers can say the work is
  drifting from notation into recognition.

## 4. Discrete and tokenized motion representations

This is the most relevant fast-moving area.

### 4.1 Sign-language side

Recent sign-language papers increasingly use learned discrete codes:

- Unsupervised discrete ASL representations tokenize fingerspelling motion into
  interpretable discrete units.
- Data-driven sign-language production work argues that classical linguistic
  resources are scarce and learns codebook-based motion representations instead.
- Signs-as-Tokens discretizes continuous signs into token sequences for
  different body parts, specifically to interface with language-model-style
  generation.
- SHuBERT shifts attention toward self-supervised transferable sign
  representations across datasets and tasks.
- M3T (2026) expands discrete sign tokenization into multimodal motion tokens
  with explicit non-manual channels, showing how fast the sign side is moving.

Concrete papers to anchor:

- "Unsupervised Discrete Representations of American Sign Language", EMNLP
  2024. Important because it explicitly trains a tokenizer for ASL
  fingerspelling and adds an interpretability-oriented loss.
- "A Data-Driven Representation for Sign Language Production", 2024.
  Important because it learns a VQ codebook of short sign motions for
  downstream production rather than using human-written notation.
- "Signs as Tokens", ICCV 2025. Important because it discretizes continuous
  signs into body-part token sequences for multilingual sign generation.
- "SHuBERT: Self-Supervised Sign Language Representation Learning", 2024/2025.
  Important because it strengthens the baseline expectation that sequence-level
  transferable sign representations should exist.
- "M3T: Discrete Multi-Modal Motion Tokens for Sign Language Production",
  2026. Important because it extends tokenization beyond manual hand channels
  to facial and other non-manual components.

Takeaway:

- "Continuous motion -> discrete symbolic / token sequence" is already a
  credible and active idea.
- Therefore, our novelty cannot just be "discretize hand motion".
- The differentiator must be interpretability and structure:
  temporal HL should be explicitly human-readable, anatomically factored, and
  evaluation-friendly in ways learned VQ tokens are not.
- The stronger contrast is not "symbolic vs tokenized", but
  "human-readable anatomical-temporal factorization vs opaque learned codebook".

### 4.2 Vision / hand / HOI side

Recent CV and HOI papers also move toward discrete representations:

- HOIGPT introduces a hand-object decomposed VQ-VAE to discretize long HOI
  sequences before language-model processing.
- TokenHand uses discrete token representation for efficient hand mesh
  reconstruction.
- Handformer2T proposes hand-level tokenization for interacting hand pose
  estimation.
- Gesture-aware pretraining and token fusion (2026) uses gesture labels as a
  supervisory prior for 3D hand pose estimation, which is another sign that
  discrete or symbolic side information is being used as a learning bias.

Takeaway:

- Tokenization is now mainstream enough that reviewers will not consider
  "discrete representation" alone sufficient novelty.
- A strong temporal HL story should emphasize that unlike learned latent tokens,
  our symbols expose controllable temporal semantics:
  direction change, persistence, opening/closing trend, cross-hand approach,
  synchronization, and possibly segment boundaries.
- The most dangerous nearby review framing is that temporal HL is just a
  hand-crafted tokenizer. We need evidence that the hand-crafted structure buys
  transferability, analysis, or controllability that latent tokens do not.

## 5. Temporal hand-pose and hand-motion modeling

Important neighboring threads:

- Hierarchical Temporal Transformer for 3D Hand Pose Estimation and Action
  Recognition (CVPR 2023) explicitly models different temporal granularities for
  pose and action.
- HandFormer treats 3D hand poses as a compact but informative temporal signal
  for action recognition and combines dense pose with sparse RGB context.
- ICCV 2025 sequential 3D hand pose work adds explicit dynamic temporal
  modeling and memory for pose estimation.
- CVPR/ICCV 2025 work on 4D hand motion recovery and generation pushes stronger
  temporal consistency, physics, diffusion, and long-sequence modeling.
- UniHand (2026) further unifies multiple controlled 4D hand motion tasks.

Concrete papers to anchor:

- Wen et al., "Hierarchical Temporal Transformer for 3D Hand Pose Estimation
  and Action Recognition", CVPR 2023.
- Shamil et al., "On the Utility of 3D Hand Poses for Action Recognition",
  ECCV 2024 / HandFormer.
- Yu et al., "Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic
  Camera", CVPR 2025.
- Zhang et al., "Diffusion-based 3D Hand Motion Recovery with Intuitive
  Physics", ICCV 2025.
- Sun et al., "UniHand: A Unified Model for Diverse Controlled 4D Hand Motion
  Modeling", 2026.

Takeaway:

- "Temporal modeling helps hands" is already known.
- "Longer sequence modeling for hand pose or hand motion" is already known.
- The remaining space is not better temporal encoders alone, but a temporal
  symbolic layer that is useful even when vision is removed.

This is exactly why the local symbolic experiments are promising:

- they already show transfer and retrieval gains from temporal HL without
  needing a new vision backbone.

Concrete review risk:

- If our strongest evidence remains vision-centric, reviewers can collapse the
  paper into the large temporal hand-modeling literature.
- The differentiator must be representation-level utility under weak or absent
  visual evidence, not just stronger temporal estimation.

## 6. Robot and control interfaces

Relevant trend:

- Recent robotics work increasingly uses compact pose or token interfaces to
  bridge perception and control.
- PoseLess maps images to hand-joint control without explicit depth-based pose
  recovery, using tokenized representations.
- Pose-VLA argues for discrete pose tokens as a universal interface between
  perception and robot action.
- Uni-Hand (forecasting + robotic imitation angle) also shows that hand motion
  abstractions can be treated as a bridge from human video to downstream robot
  behavior.

Takeaway:

- The application claim "symbolic / tokenized hand representation can be used as
  an intermediate control interface" is now credible.
- But it is also no longer unique.
- For us, robot control should stay as a downstream validation target, not the
  core novelty claim.

Concrete review risk:

- If robotics becomes the headline, the work becomes vulnerable to stronger
  robot-policy/token-interface papers.
- If robotics stays downstream, it instead strengthens the argument that
  temporal HL is a reusable intermediate representation.

## 7. What space is already occupied

The following contribution shapes are weak or already crowded:

- Better frame-wise HL recognizer only
- Temporal transformer on top of old HL labels only
- Learned hand-motion tokens without strong interpretability
- Generic sign-language-style symbolic encoding claim
- "Can be used for robotics" as the main contribution

## 8. What space still looks open

The strongest open niche is:

- an interpretable sequence-native symbolic representation for generic hand
  articulation
- with explicit temporal factors, not just per-frame symbols
- derived automatically from 3D hand motion
- shown to carry reusable structure across tasks
- and evaluated both intrinsically and in downstream transfer

More concretely, the open representation factors are:

- transition type between adjacent frames
- duration / persistence of stable states
- hand-level motion phase
- finger-group coordination
- cross-hand relation dynamics
- optional temporal segment boundaries

This is meaningfully different from:

- frame-wise HL
- learned latent tokenizers with opaque codebooks
- task-specific temporal hand estimators
- sign-language notation systems centered on linguistic meaning

Refined formulation of the open niche:

- Not "a better tokenizer for hand motion"
- Not "a temporal network for HL"
- But "a human-readable, sequence-native symbolic hand representation whose
  temporal factors are explicit and whose utility survives outside the original
  image-to-notation task"

## 9. Direct implications for our current local results

What already aligns well:

- The deterministic temporal HL construction is a valid first step because it
  turns time into symbolic channels instead of mere temporal context.
- Symbolic retrieval gains suggest temporal channels preserve action identity.
- Pretrain -> finetune transfer gains suggest temporal HL carries reusable
  structure beyond a single label space.
- State-only HL being less stable than temporal HL is an important signal that
  time-aware symbolic structure is not just a convenience feature.

What is still missing for a top-tier submission:

- stronger intrinsic metrics for sequence-native quality
- direct comparison against alternative discrete representations
- a student model showing temporal HL helps raw-joint or image-based learning
- stronger evidence on duration / segmentation / coordination, not only
  adjacent-frame transition labels
- a clearer proof that temporal HL is better because of representation, not just
  because of extra supervised channels

## 11. Closest threat models as of 2026-06-09

If reviewers are tough, the nearest external-paper attack lines are likely:

1. "This is just frame-wise HL plus temporal modeling."
   Response path:
   only solvable if the label space itself includes temporal factors and those
   factors show measurable standalone utility.

2. "This is just a hand-crafted alternative to VQ tokens."
   Response path:
   only solvable if temporal HL beats or complements learned-token baselines on
   transfer, retrieval, controllability, or diagnosis.

3. "This belongs to sign-language tokenization / notation instead."
   Response path:
   only solvable if the task remains generic hand articulation, avoids semantic
   sign assumptions, and demonstrates coverage outside sign language.

4. "This is just another temporal hand-pose representation."
   Response path:
   only solvable if the representation remains useful after removing the vision
   model and can be manipulated, compared, or transferred symbolically.

## 12. Working literature shortlist for our project

This is the minimum set that should be actively cited in all internal planning:

- HL: ACM MM 2024 Hand Labanotation
- Labanotation automation: Cai et al. 2023; Jiang et al. 2024; LabanLab 2025
- Sign notation systems: HamNoSys; SignWriting
- Sign tokenization / representation:
  EMNLP 2024 discrete ASL, Data-Driven SLP 2024, Signs-as-Tokens 2025,
  SHuBERT 2024/2025, M3T 2026
- Temporal hand modeling:
  HTT 2023, HandFormer 2024, Dyn-HaMR 2025, DIP-Hand 2025, UniHand 2026
- Robot/control token interfaces:
  PoseLess 2025, Pose-VLA 2026

## 10. Recommended literature-facing experiment agenda

Priority order:

1. Finish the raw-joint student experiment and test whether temporal HL teacher
   supervision improves sequence recognition over direct joint windows.
2. Add sequence-native intrinsic metrics:
   edit distance, transition F1, run-length error, event/boundary F1 if segment
   labels are added.
3. Add at least one stronger temporal representation ablation:
   state-only HL
   state+transition HL
   state+transition+duration HL
4. Add one learned-token control if feasible, even a lightweight VQ baseline on
   joint windows, to show interpretability is not coming for free.
5. Keep robotics as optional downstream validation, not the central story.

## 11. Bottom-line positioning

The most defensible project identity is:

- not a better hand-pose estimator
- not a sign-language notation system
- not a generic motion tokenizer
- not a robot policy paper

Instead:

- a sequence-native symbolic hand representation project
- with explicit temporal structure
- whose value is validated by transfer, retrieval, and possibly reconstruction /
  control
- and whose interpretability is a first-class property rather than a byproduct

## 12. Concrete paper-risk warnings for later experiment design

These arguments are likely to appear in reviews if we do not preempt them:

- "This is just temporal modeling on top of HL."
- "Why not use a learned tokenizer instead of hand-designed symbols?"
- "This seems too close to sign-language notation."
- "If vision is not central, why is this not just another action-recognition
  feature engineering paper?"
- "Where is the evidence that the temporal symbolic layer itself, rather than
  extra labels, is the reason for the gains?"

So the experiment plan should explicitly answer them.

## 13. Concrete 2023-2026 anchor papers to compare against

This section is a dated reading map for experiment design. It is still a
research memo, not paper text.

### 13.1 Direct ancestor and closest neighbors

- Hand Labanotation (ACM MM 2024): the direct parent work. Core scope is still
  frame-wise HL symbols plus image-to-symbol translation on HLD.
- Hierarchical Temporal Transformer for 3D Hand Pose Estimation and Action
  Recognition (CVPR 2023): strong evidence that multi-granularity temporal
  modeling is useful for hand pose and action, but the target remains pose /
  action, not symbolic hand-motion documentation.
- Prior-aware Dynamic Temporal Modeling Framework for Sequential 3D Hand Pose
  Estimation (ICCV 2025): reinforces that sequential hand pose estimation is an
  active area and that explicit temporal memory improves smoothness and
  robustness under occlusion.
- UniHand (2026): shows that "4D hand motion" is now a recognized umbrella task
  spanning estimation and generation, so a temporal HL upgrade must position
  itself as a representation layer rather than another motion model.

Implication:

- We cannot sell "temporal modeling for hand motion" by itself.
- We must sell "sequence-native symbolic representation for hand motion".

### 13.2 Discrete / tokenized hand or pose representations

- HandFormer (ECCV 2024): argues that 3D hand poses are a compact but
  informative temporal modality for action recognition.
- Handformer2T (WACV 2024): uses hand-level tokenization for interacting hand
  pose estimation from RGB.
- HOIGPT (CVPR 2025): discretizes long hand-object interaction sequences with a
  hand-object decomposed tokenizer before language-model processing.
- TokenHand (CVPR 2026): uses discrete tokens for efficient hand mesh
  reconstruction.
- UniPose (CVPR 2025): treats pose tokens as a unified interface for pose
  comprehension and generation.

Implication:

- Reviewers have already seen strong tokenization stories.
- Our differentiator must be explicit interpretability and temporal semantics,
  not "we also discretize motion".

### 13.3 Sign-language symbolic and discrete representation work

- Ham2Pose (CVPR 2023): animates HamNoSys notation into sign pose sequences.
- HamNoSys-based Motion Editing Method for Sign Language (LREC-COLING Workshop
  2024): uses HamNoSys as an editing interface over MoCap-driven sign motion.
- Unsupervised Discrete Representations of American Sign Language (EMNLP 2024):
  learns discrete units for ASL fingerspelling.
- SHuBERT (2024 preprint): self-supervised sign representation learning with
  contextual multi-stream cluster prediction.
- Signs as Tokens / SOKE (ICCV 2025): discretizes signs into token sequences for
  multilingual sign generation.

Implication:

- The sign-language community already treats continuous motion as discrete
  symbolic or token sequences.
- Temporal HL must stay centered on generic hand articulation rather than
  linguistic semantics, otherwise the work will be pulled into the wrong
  comparison space.

### 13.4 Automatic Labanotation generation and symbolic motion interfaces

- LabanFormer (2023): transformer-based Labanotation generation for dance /
  skeletal motion.
- Automatic generation of Labanotation based on human pose estimation (2023):
  another body-motion pipeline from video / pose to notation.
- Automatic generation of Labanotation based on a hybrid transformer-LSTM
  network with multi-scale spatio-temporal features (Scientific Reports 2026):
  confirms the full-body dance line is still active.
- LaMoGen (CVPR 2026): uses a Labanotation-inspired symbolic motion interface
  for text-to-motion generation.

Implication:

- There is active symbolic-motion work, but it is dominated by full-body dance
  or text-to-motion.
- A hand-specific, anatomically factored, sequence-native symbolic layer still
  appears underexplored.

### 13.5 Robotics and control-interface line

- PoseLess (2025): maps monocular images directly to hand-joint control using a
  tokenized representation, avoiding explicit pose estimation.
- Pose-VLA (2026): argues for discrete pose tokens as a universal interface
  between perception and robot actions.
- FAST / FAST+ (2025): shows action tokenization is becoming central in
  high-frequency robot control.

Implication:

- "Compact symbolic / token interface for control" is credible but crowded.
- Robotics should remain downstream validation, not the main novelty claim.

## 14. Updated reading-based project filter

After anchoring the nearby literature, the strongest and weakest project shapes
are clearer.

### Weak project shapes

- image/video -> better frame-wise HL predictor
- temporal transformer over old HL labels only
- learned VQ tokens without interpretability
- sign-language-style notation claim
- robot-control interface as the headline contribution

### Strong project shape

- explicit temporal factors inside the representation itself
- representation usable without the original image backbone
- evidence that the symbolic sequence is reusable across tasks
- clear intrinsic sequence metrics beyond plain per-frame accuracy

## 15. Immediate experiment consequences

The local results already support part of this positioning:

- symbolic temporal HL benefits from transfer more reliably than state-only HL
- pretrain-only time normalization helps symbolic and joint-sequence branches
- the strongest gains concentrate on difficult subsets like wrist ROM,
  occlusion, and interaction-heavy cases

What still needs to be built to match the literature-facing gap:

- duration / persistence channel, not only adjacent-frame transition labels
- stronger sequence-native intrinsic metrics
- at least one lightweight learned-token control baseline
- at least one stronger student or downstream test where temporal HL wins as a
  representation, not only as a label space

## 16. External verification pass on 2026-06-09

This section records an explicit web check performed on 2026-06-09 so that
later experiment planning is anchored to confirmed nearby literature rather
than memory.

### 16.1 Direct ancestor

- Hand Labanotation is indeed the ACM MM 2024 paper "Translating Motion to
  Notation: Hand Labanotation for Intuitive and Comprehensive Hand Movement
  Documentation", released via OpenReview / ACM DL.
- Its central artifacts are the Hand Labanotation symbol system, the HLD
  dataset, and the MHLFormer image-to-framewise-HL translator.

Implication:

- A project framed as "better MHLFormer" or "HL with a temporal encoder" will
  look obviously incremental against the actual 2024 baseline.

### 16.2 Full-body Labanotation automation is active

- Cai et al. 2023 (Neural Computing and Applications) explicitly performs
  automatic Labanotation generation for folk dance videos via pose estimation
  and key-frame extraction.
- Jiang et al. 2024 (IJCNN) studies motion part-level interpolation and
  manipulation on top of automatic symbolic Labanotation annotation.
- LabanLab (Eurographics 2025 short paper) turns Labanotation into an
  interactive choreography interface with motion preview.
- LaMoGen / LabanLite (OpenReview 2026) pushes Labanotation-inspired symbolic
  motion representation into language-to-motion generation.

Implication:

- "Notation is useful for motion interfaces" is already established.
- The hand-specific opportunity is narrower: anatomically fine-grained,
  sequence-native hand articulation symbols.

### 16.3 Sign notation and sign-sequence modeling are strong comparison lines

- HamNoSys remains the canonical phonetic sign transcription system centered on
  handshape, orientation, location, movement, and non-manual features.
- SignWriting explicitly covers handshape, movement, and facial components for
  writing signed languages.
- Ham2Pose (2023) animates HamNoSys notation into sign pose sequences.
- A HamNoSys-based motion editing method appeared in the 2024 sign-language
  workshop literature.

Implication:

- If temporal HL drifts toward linguistic semantics, reviewer framing will
  switch from generic hand articulation to sign-language phonology.
- Keep the scope on non-semantic, generic hand articulation and downstream
  transfer/control utility.

### 16.4 Discrete-token literature is no longer optional background

- EMNLP 2024 "Unsupervised Discrete Representations of American Sign Language"
  learns a tokenizer for ASL fingerspelling.
- "A Data-Driven Representation for Sign Language Production" (2024) learns a
  VQ codebook of short sign motions and translates text into token sequences.
- SHuBERT (ACL 2025) learns contextual sign representations from about 1,000
  hours of ASL video via self-supervised multi-stream cluster prediction.
- Signs as Tokens / SOKE (ICCV 2025) discretizes continuous signs into
  body-part token sequences for multilingual sign generation.
- M3T (2026) extends discrete sign motion tokens to multi-modal manual and
  non-manual channels.
- HOIGPT (CVPR 2025) uses a hand-object decomposed VQ tokenizer for long HOI
  sequences.
- TokenHand (CVPR 2026) uses discrete tokens for efficient hand mesh
  reconstruction.

Implication:

- The novelty target cannot be "we discretize hand motion".
- The key contrast must be "human-readable, anatomically factored, temporally
  explicit symbols" versus opaque learned codes.

### 16.5 Temporal hand modeling is also crowded

- HTT (CVPR 2023) already shows hierarchical temporal modeling for 3D hand pose
  estimation and action recognition.
- HandFormer (ECCV 2024) treats dense 3D hand pose sequences as a compact
  temporal modality for action recognition.
- Prior-aware Dynamic Temporal Modeling (ICCV 2025) strengthens the sequential
  hand-pose estimation line under temporal smoothness and occlusion concerns.
- Dyn-HaMR (CVPR 2025) reconstructs 4D interacting hand motion from dynamic
  monocular video.
- UniHand (2026) pushes toward unified controlled 4D hand motion modeling.

Implication:

- "Temporal modeling helps hands" is established.
- We need evidence that temporal HL is valuable as a representation layer even
  outside the original vision task.

### 16.6 Dataset landscape changed after the original HL paper

- InterHand2.6M remains the standard large-scale multiview interacting-hand
  dataset from ECCV 2020.
- FreiHAND remains the standard RGB single-hand pose/shape dataset from ICCV
  2019.
- GigaHands (CVPR 2025) now offers a large bimanual activity dataset with
  sequence-level activity coverage and richer downstream potential.
- SignAvatars (ECCV 2024) provides a large 3D sign-language holistic motion
  dataset with mesh annotations.

Implication:

- A future temporal-HL paper cannot present "large temporal hand sequences" as
  a novelty by itself.
- The novelty must be in the representation and the evidence, not only in data
  scale.

## 17. Updated reviewer threat ranking after the external pass

The likely attack lines are now clearer:

1. "This is only frame-wise HL plus sequence modeling."
2. "This is just a hand-crafted tokenizer, while learned discrete tokens are
   more flexible."
3. "This belongs to sign notation / sign generation rather than generic hand
   representation."
4. "This is another temporal hand-pose modeling paper."

The experiments most directly aligned to these risks are:

- symbolic-vs-learned-token control
- state vs state+transition vs state+transition+duration ablation
- retrieval / transfer / student evidence without relying on the same vision
  head
- at least one sequence-native intrinsic metric beyond frame accuracy

## 18. Additional 2026 threat update: symbolic motion is accelerating

This section records a second web pass on 2026-06-09 focused on the newest
symbolic-motion and discrete-token papers that are most dangerous to a temporal
HL project.

### 18.1 Broad symbolic human-motion representation is becoming explicit

- LingoMotion (arXiv 2026) proposes an interpretable and unambiguous symbolic
  representation for human motion built around joint-angle-based "letters",
  "words", and "phrases".
- LaMoGen / LabanLite (CVPR 2026) pushes a Labanotation-inspired symbolic
  interface into language-to-motion generation and explicitly sells
  interpretable symbolic reasoning as the interface between language and motion.

Implication:

- The field is no longer satisfied with only opaque latent motion tokens.
- But this also means "interpretable symbolic motion" is no longer unique.
- The hand-specific win condition must therefore be narrower and sharper:
  anatomically grounded, sequence-native hand articulation symbols with clear
  temporal semantics and measurable utility outside generation.

### 18.2 Hand-side discrete tokenization is now fully mainstream

- TokenHand (CVPR 2026) reformulates monocular 3D hand mesh reconstruction as
  discrete token classification rather than direct continuous regression.
- This makes the "discrete hand representation" claim even less defensible as a
  novelty target by itself.

Implication:

- We should avoid any framing that sounds like "we discretize hand pose / hand
  motion better".
- The contrast must remain:
  human-readable temporal hand symbols versus efficient but opaque learned hand
  tokens.

### 18.3 Sign representation moved further toward large-scale transferable units

- SHuBERT appeared in ACL 2025 as a self-supervised contextual sign
  representation model trained on about 1,000 hours of ASL video.
- M3T (2026) extends discrete sign production to multi-modal motion tokens,
  explicitly spanning hands, face, and body-side channels.

Implication:

- Sequence-level transferable discrete representations are now expected in
  nearby sign-language work.
- Temporal HL should not compete on "largest pretraining scale" or "best
  generic tokenizer".
- Its advantage must come from representation structure:
  explicit anatomical factorization, temporal interpretability, and direct
  diagnostic / retrieval / control value.

### 18.4 Dataset novelty is getting squeezed further

- GigaHands (CVPR 2025 highlight) expands the bimanual sequence-data landscape
  with activity-level coverage, object context, and downstream generation
  utilities.

Implication:

- A future temporal-HL submission cannot rely on "we also build temporal hand
  sequences" as a headline.
- Dataset work is only defensible if the new annotations expose sequence-native
  symbolic structure that current datasets do not already provide.

## 19. Updated hard filter for our project after the second pass

After including the 2026 symbolic-motion papers, the project filter is stricter
than before.

### 19.1 Still too weak

- HL plus any stronger temporal encoder
- a generic discrete tokenizer for hand motion
- a broad "interpretable motion language for hands" story without hard
  sequence-level evidence
- a dataset-scale story without representation-level novelty

### 19.2 Still strong enough to pursue

- a sequence-native symbolic hand representation whose temporal variables are
  explicit in the notation itself
- a representation that stays useful when detached from the original image
  recognizer
- a hand-specific symbolic layer that beats or complements learned tokens in at
  least one concrete regime:
  low-data transfer, retrieval, diagnostic controllability, or robot-interface
  stability

### 19.3 Concrete design consequence

If we keep using the name "temporal HL", the representation upgrade needs to
contain at least one of the following beyond adjacent-frame transition:

- duration / persistence
- phase / sub-action progress
- coordinated finger-group events
- cross-hand relation events
- symbolic segment boundaries

Otherwise the work remains exposed to the simplest review attack:
"this is just HL with sequence context."

## 20. Synthesis after combining external literature with local evidence

This section merges the external verification pass with the current local
ablation results so later work does not drift back into already-weakened
directions.

### 20.1 What the literature says is crowded

- frame-wise HL translation is already owned by the ACM MM 2024 parent paper
- temporal hand modeling is already crowded across pose estimation, action
  recognition, and 4D motion recovery
- learned discrete motion / pose tokenization is now standard enough that
  "discretize hand motion" is not a novelty claim
- broad symbolic-motion stories are no longer unique after 2026 works like
  LingoMotion and LaMoGen / LabanLite

### 20.2 What the local experiments already rule out

- `HL + temporal encoder` is not strong enough as the project identity
- handcrafted duration is not a reliable additive gain in the current setup
- exact-state event augmentation is a negative control
- current global coordination bag-of-stats is also a negative control
- generic family-reranking on the current correction benchmark is not a useful
  mainline because the audited disagreement set is degenerate

### 20.3 What currently survives both filters

The strongest remaining position is still:

- a sequence-native symbolic hand representation
- whose temporal content is explicit inside the representation rather than only
  in the model
- whose strongest currently validated temporal factor is `transition`
- and whose value is demonstrated through representation-level utility such as
  low-data transfer, symbolic retrieval, diagnostic analysis, or controlled
  correction

### 20.4 What must be true for a strong AAAI-2027-grade story

At least one of the following has to become true with evidence:

- a nontrivial temporal factor beyond raw transition helps when formulated in a
  better way than the current failed controls
- the symbolic representation clearly outperforms or complements learned-token
  baselines in a regime reviewers will care about
- the symbolic layer enables a benchmark or evaluation protocol that opaque
  tokenizers do not support cleanly
- the representation remains useful after detaching from the original image
  recognizer, e.g. through transfer, retrieval, correction, or reconstruction

### 20.5 Current best literature-facing one-sentence positioning

Not:

- a better HL recognizer
- a temporal encoder for HL
- a hand-motion tokenizer

But:

- an anatomically factored, sequence-native symbolic representation for generic
  hand articulation, where temporal variables are explicit and measurably useful
  beyond the original image-to-notation task
