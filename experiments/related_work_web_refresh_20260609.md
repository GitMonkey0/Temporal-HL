# Web-Refreshed Related-Work Memo for Temporal Hand Representation

Date: 2026-06-09

This is a research memo for experiment planning.
It is not paper text.

## Scope

Refresh the external literature around a temporal upgrade of Hand Labanotation
using publicly accessible primary sources available as of 2026-06-09.

## Executive conclusion

The crowded directions are now very clear:

- frame-wise HL recognition
- generic temporal hand modeling
- generic discrete tokenization of hand/sign motion
- symbolic motion as a bridge for generation in full-body settings

The defensible space remains:

- sequence-native symbolic hand representation
- explicit temporal channels inside the representation itself
- anatomy-aware factorization instead of monolithic codes
- human-readable local editing / repair / control
- explicit interacting-hand transition structure

## 1. Direct parent: Hand Labanotation

### Hand Labanotation

- "Translating Motion to Notation: Hand Labanotation for Intuitive and
  Comprehensive Hand Movement Documentation", ACM MM 2024.
- Establishes:
  - a 26-symbol hand-space inventory
  - HLD with more than 4M annotated images
  - MHLFormer for image-to-frame HL translation

Implication:

- A follow-up centered on better HL recognition is too weak.
- A temporal encoder over the original frame labels is likely incremental.
- The representation itself has to absorb temporal semantics.

## 2. Symbolic motion is active, but not hand-articulation-native

### Body-motion Labanotation pipelines

- Cai et al., Neural Computing and Applications 2023:
  automatic labanotation generation from folk-dance videos.
- LabanLab, Eurographics 2025:
  interactive Labanotation authoring with motion preview.
- LaMoGen, CVPR 2026:
  language-to-motion generation through symbolic LabanLite inference.

Implication:

- Symbolic motion representation is already validated as a useful interface.
- But these works are body-motion oriented and do not expose finger-level
  articulation and interacting-hand symbolic structure as first-class objects.
- "Notation is useful" is established; "hand-specific temporal symbolic
  structure" is still open.

## 3. Sign-language notation is adjacent, but not the right identity

### Established notation systems

- HamNoSys:
  a phonetic transcription system for sign languages.
- SignWriting:
  a writing system for sign languages used across multiple sign languages.

Implication:

- These systems justify symbolic movement description in principle.
- But they are organized around signed-language phonetics/writing rather than
  generic non-semantic hand articulation.
- If our framing drifts toward linguistic phonology, the work can be
  reinterpreted as sign-language notation rather than a general hand
  representation paper.

Safe boundary:

- stay anatomy- and motion-grounded
- keep non-semantic control / editing / documentation central

## 4. Discrete tokenization is now a major novelty threat

### Sign / multimodal tokenization

- SHuBERT, ACL 2025:
  self-supervised multi-stream sign representation learning via cluster
  prediction over hand/face/body streams.
- Signs as Tokens, ICCV 2025:
  multilingual sign generation using discretized body-part token streams.

### Hand-specific tokenization

- TokenHand, CVPR 2026:
  discrete token representation for efficient hand mesh reconstruction.

Implication:

- "We discretize hand motion" is no longer novel.
- "We tokenize hand motion for sequence modeling" is also not enough.
- The contrast must be semantic transparency and controllable structure:
  - explicit symbolic meaning per channel
  - anatomy-aware factorization
  - local editability
  - interpretable repair
  - transition-aware motifs

## 5. Generic temporal hand modeling is crowded

### Recognition / modeling / recovery

- HandFormer, ECCV 2024:
  high-temporal-resolution 3D hand-pose modeling with sparse RGB for action
  recognition.
- Dyn-HaMR, CVPR 2025:
  4D interacting hand motion recovery from monocular dynamic-camera video,
  including interacting-hand priors and occlusion handling.
- UniHand, ICLR 2026:
  unified controlled 4D hand motion modeling under diverse conditions.

Implication:

- "Temporal hand modeling" by itself is not novel enough.
- "Temporal consistency" and "sequence understanding" are baseline terrain.
- To stand out, the temporal component has to live in the representation and
  enable tasks these pipelines do not naturally expose.

## 6. 2026 frontier pressure is specifically about dexterous interaction

### Generation / interaction pressure

- HandX, CVPR 2026:
  argues whole-body models miss fine-grained cues such as finger articulation,
  contact timing, and inter-hand coordination; builds a bimanual dexterous
  motion foundation.
- TSHaMo, arXiv 2026:
  text-driven 3D hand motion generation, motivated by the lack of hand-specific
  motion generation methods beyond full-body or object-dependent settings.
- Text-Driven 3D Hand Motion Generation from Sign Language Data, CVPR 2026:
  open-vocabulary language-conditioned 3D hand motion generation from sign data.

Implication:

- The field is explicitly moving toward hand-specific temporal structure.
- Inter-hand coordination and transition timing are no longer side topics.
- Any next-step hand representation that leaves interaction dynamics implicit is
  likely underpowered.

## 7. Positioning constraints for our project

What looks weak:

- old HL + temporal backbone
- sequence classification as the main win
- generic tokenization / compression framing
- "symbolic because interpretable" without a realized control task

What still looks strong:

- sequence-native symbolic hand representation
- explicit state / transition / coordination factorization
- anatomy-aware grouped structure
- local edit / repair / donor transfer tasks
- interacting-hand transition reasoning

## 8. Concrete experiment pressure implied by the literature

Any strong submission now likely needs evidence for:

- benefit beyond frame-wise HL
- benefit beyond stronger temporal encoders on old labels
- local controllability advantages over opaque tokens
- explicit transition-aware tasks, not only state prediction
- robustness on interacting-hand slices where current geometric or learned-token
  pipelines remain hard to inspect and edit

## 9. Practical bottom line

As of 2026-06-09, the safest mainline is still:

- treat Temporal HL as a representation upgrade
- make transition channels first-class
- keep grouped anatomy explicit
- make interaction-aware editing the hardest proving ground
- avoid turning the project into another tokenization or temporal-modeling paper
