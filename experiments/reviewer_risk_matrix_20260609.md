# Reviewer Risk Matrix for Temporal HL

Date: 2026-06-09

This file is a planning memo, not paper text.

## Purpose

Map likely reviewer attacks to the neighboring literature and define what
evidence is required to survive each attack.

## Reading rule

For each risk:

- `collision` means the nearby work family that can collapse our novelty
- `failure mode` means what would make our project look incremental
- `required counter-evidence` means what experiments or artifacts we need

## Risk 1: "This is just HL plus a temporal model"

### Collision

- Hand Labanotation, ACM MM 2024
- generic temporal encoders for hand modeling

### Failure mode

- Keep the old frame-wise HL label space unchanged
- Add a temporal transformer / sequence loss / temporal smoothing
- Report only better recognition or retrieval

### Required counter-evidence

- explicit new temporal symbol channels in the representation
- ablation:
  - old HL + stronger temporal encoder
  - Temporal HL + matched encoder
- task that directly consumes temporal labels:
  - transition-conditioned edit
  - transition retrieval
  - event prediction

### Current local state

- positive:
  - transition-conditioned audits already exist
  - donor-only oracle failure already shows current-frame state is insufficient
- add:
  - the matched old-HL vs temporal-HL report now shows that temporal HL does
    **not** win under the same strong sequence-classification protocol
  - this removes the weak fallback story that temporal HL is merely easier for
    the same encoder to classify
- remaining gap:
  - interaction-aware realized editing is still not fully solved, so the
    strongest temporal claim must remain on control/editability/search rather
    than universal editing quality

## Risk 2: "This is just another motion tokenizer"

### Collision

- sign tokenization papers
- HOIGPT
- TokenHand
- broader VQ / learned discrete motion literature

### Failure mode

- Sell the work as discrete coding, compression, or sequence modeling
- Use learned tokens as the central object rather than explicit symbols
- Evaluate mostly with classification / retrieval scores

### Required counter-evidence

- explicit symbolic semantics per channel
- local edit audit showing symbolic locality
- family-level repair or interpretable correction analysis
- controlled edit / transplant / counterfactual tasks where opaque proxies show
  collateral damage

### Current local state

- positive:
  - strong local edit evidence
  - zero-harm family repair
  - counterfactual consistency and conditional transplant audits
- missing:
  - a stronger temporal-control experiment that realizes edited motion rather
    than only auditing donor quality

## Risk 3: "Temporal hand modeling is already crowded"

### Collision

- HandFormer
- Dyn-HaMR
- UniHand
- related 4D hand-motion and action pipelines

### Failure mode

- claim novelty on temporal consistency, sequence modeling, or better hand
  motion understanding in general
- rely on continuous geometry reconstruction framing

### Required counter-evidence

- keep representation-level temporal semantics central
- show a task continuous pipelines do not naturally expose:
  - symbolic editability
  - symbolic repair
  - explicit transition motifs
  - controllable local manipulation

### Current local state

- positive:
  - hand-motion requires transition channels is already strongly supported
  - hard single-hand slices are largely solved by transition-conditioned motifs
- missing:
  - interaction-aware realized editing remains unsolved

## Risk 4: "This belongs to sign-language notation, not generic hand motion"

### Collision

- HamNoSys
- SignWriting
- sign production pipelines with discrete sub-units

### Failure mode

- center the problem on semantic gesture classes
- adopt linguistic terminology as the main decomposition
- ignore non-semantic hand articulation use cases

### Required counter-evidence

- keep labels grounded in anatomy and motion, not language units
- evaluate generic hand-motion control / retrieval / repair tasks
- avoid relying on sign-semantic downstreams as the main proof

### Current local state

- positive:
  - current audits are non-semantic and anatomy-grounded
- add:
  - [related_work_deep_refresh_20260609.md](/opt/tiger/hand/experiments/related_work_deep_refresh_20260609.md)
    now makes the identity boundary explicit:
    - HamNoSys / SignWriting are adjacent
    - Temporal HL must stay non-semantic and anatomy-grounded
- remaining gap:
  - this boundary is now documented internally, but future experiment memos
    should keep using the same terminology consistently

## Risk 5: "Body-motion Labanotation already did symbolic editing / generation"

### Collision

- automatic body Labanotation generation
- symbolic motion manipulation and Labanotation preview systems
- language-to-Labanotation-to-motion pipelines

### Failure mode

- frame the contribution as "notation enables manipulation"
- do not explain why fingers and interacting hands require a different
  representation granularity

### Required counter-evidence

- show hand-specific temporal bottlenecks:
  - dense short-term transitions
  - local articulation dependencies
  - inter-hand coordination
- compare state-only and transition-aware symbolic formulations

### Current local state

- positive:
  - transition-conditioned motif evidence already separates hand-motion from
    frame-state substitution
- add:
  - the interaction-vs-noninteraction summary bundle now provides a concise
    top-level view of where symbolic grouped-motif editing is strong and where
    interaction remains weak
- remaining gap:
  - interaction-aware realized editing is still the main unsolved issue, not
    the existence of a summary artifact

## Risk 6: "The symbolic advantage is really just post-hoc repair"

### Collision

- skeptical reviewer reading family-repair gains as engineering cleanup

### Failure mode

- over-index on grouped+family repaired numbers
- fail to separate intrinsic structure from repair

### Required counter-evidence

- keep structure and repair as distinct contributions
- report:
  - grouped-vs-flat gains
  - family-repair zero-harm band
  - editability benefits that exist even before repair

### Current local state

- positive:
  - structural frontier tables already separate grouped from family-repair
- add:
  - the new compact hard-slice search bundle cleanly isolates a
    control-oriented temporal mechanism rather than repair cleanup
- add:
  - the new topline evidence bundle now merges:
    - structure
    - repair
    - control
    - temporal transition
    - hard-slice compact search
- remaining gap:
  - only the final presentation polish remains, not the underlying summary
    coverage

## Risk 7: "The remaining hard cases are interaction failures, so the method is not mature"

### Collision

- direct weakness in current local results

### Failure mode

- present transition-conditioned hand-motion as a clean universal solution
- underplay the interaction slice weakness

### Required counter-evidence

- explicitly limit the claim:
  - single-hand hard slices are largely solved
  - interaction-aware motion editing remains the main open problem
- add a next-stage experiment focused on interaction-aware temporal editing

### Current local state

- positive:
  - slice audits already isolate interaction as the main weakness
- add:
  - the hard-slice compact-search bundle now summarizes the concentrated
    right-hand interaction bottlenecks and the compact mechanism that addresses
    them
- add:
  - the topline evidence bundle now makes the hard-slice boundary explicit at
    the same level as structure/control/transition evidence
- add:
  - the new interaction-vs-noninteraction summary now exposes the split between
    easier noninteraction edits and harder interaction edits at the same
    top-level layer
- remaining gap:
  - the method is still not mature enough to claim globally solved interaction
    editing

## Risk 8: "There is no downstream reason to prefer the representation"

### Collision

- any reviewer who accepts notation but asks why it matters

### Failure mode

- evidence stops at intrinsic retrieval and classifier transfer
- no realized control or reconstruction advantage

### Required counter-evidence

- at least one downstream where explicit temporal symbols are the natural
  interface:
  - transition-conditioned editor
  - motion reconstruction from temporal symbols
  - robot-control style smoothing / editability

### Current local state

- positive:
  - strong audit evidence for control-like locality
- missing:
  - a realized downstream mechanism rather than donor or retrieval analyses

## Prioritized action list

Highest priority:

1. Build a transition-conditioned symbolic editor or simulator.
2. Add an interaction-aware edit benchmark because interaction is the dominant
   remaining weakness.
3. Use the new topline evidence bundle as the default internal decision entry
   point and keep it updated as new strong results land.

Second priority:

4. Add an explicit old-HL-plus-strong-encoder comparison centered on the new
   temporal channels.
5. Add an interaction-vs-noninteraction summary table.

Lower priority:

6. More raw classifier tuning.
7. More token-proxy discrimination contests.
