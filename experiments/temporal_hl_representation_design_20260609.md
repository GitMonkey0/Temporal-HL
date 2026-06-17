# Temporal HL Representation Design Memo

Date: 2026-06-09

This file is a design memo for experiments and label construction.
It is not paper text.

## Goal

Turn the current strategic conclusion into concrete representation constraints:

- what Temporal HL should encode
- what it should not encode
- which channels are mainline vs auxiliary
- which experiments each channel enables

## Core design rule

Temporal HL must be a representation upgrade, not an encoder upgrade.

That means:

- at least one temporal concept must be explicit in the labels
- downstream tasks must directly use that temporal concept
- if the temporal concept can be removed without changing the task, the design
  is too weak

## Design target

Represent hand motion as a factored symbolic object with three levels:

1. `state`
2. `transition`
3. `coordination`

These should remain explicit and anatomically grounded.

## Level 1: state channels

These preserve continuity with frame-wise HL and should remain readable.

### S1. Regional direction state

- keep the original hand-part directional state for local anatomy
- this remains the anchor for readability and backward compatibility

Status:

- mainline

### S2. Hand-level state signatures

- grouped summaries such as left-hand state, right-hand state, and coarse
  interaction state

Status:

- mainline for analysis and retrieval
- useful for grouped factorization and family repair

## Level 2: transition channels

This is the actual temporal core. At least one of these channels must be
first-class in the final representation.

### T1. Per-part transition magnitude

- example buckets:
  - `stay`
  - `minor_shift`
  - `major_shift`

Why it matters:

- provides the minimal temporal upgrade over frame-wise HL
- supports transition-conditioned matching and motif retrieval

Status:

- mainline

### T2. Hand-motion trend

- example buckets:
  - `opening`
  - `closing`
  - `mixed`
  - `stable`

Why it matters:

- aligns with current local evidence that hand motion is transition-aware
- directly supports control-like edit tasks

Status:

- mainline

### T3. Interaction-motion trend

- example buckets:
  - `approach`
  - `separate`
  - `stable`
  - optional `contact-maintain`

Why it matters:

- current evidence already shows interaction motion is one of the cleanest
  editability wins
- also exposes the major remaining weakness under interacting-hand settings

Status:

- mainline

### T4. Persistence / dwell

- example buckets:
  - short
  - medium
  - long

Why it matters:

- turns repeated state into an explicit symbolic fact
- can help segmental reasoning and temporal compression

Risk:

- local evidence says persistence is useful but not yet a stable mainline win

Status:

- auxiliary for now

### T5. Segment boundary / phase

- example buckets:
  - onset
  - hold
  - reversal
  - release

Why it matters:

- gives stronger temporal abstraction than adjacent-frame transitions
- could support edit scripts and motif composition

Risk:

- current local evidence says coarse phase compression loses margin
- refined phase helps but still trails the strongest mainline

Status:

- auxiliary / exploratory

## Level 3: coordination channels

These channels make the representation sequence-native rather than a bag of
independent part transitions.

### C1. Finger-group coordination

- whether neighboring fingers move together or diverge

Value:

- may capture anatomically meaningful motifs
- aligns with grouped factorization

Risk:

- not yet clearly required by local evidence

Status:

- exploratory

### C2. Cross-hand coordination

- whether two hands change synchronously, asynchronously, or antagonistically

Value:

- likely necessary for interacting-hand editing
- directly targets the main remaining weakness

Status:

- high-priority exploratory channel

## Mainline representation recommendation

If we had to lock the representation today, the strongest mainline is:

- state:
  - regional direction state
  - grouped hand state signatures
- transition:
  - per-part transition magnitude
  - hand-motion trend
  - interaction-motion trend

Do not center the first strong version on:

- persistence as the main novelty
- coarse segment compression
- learned motif codebooks

Those can remain auxiliary branches.

## Recommended label factorization

Use a factored label space rather than one monolithic token.

Preferred factorization:

- left hand:
  - state
  - motion trend
- right hand:
  - state
  - motion trend
- pair:
  - interaction trend
- local parts:
  - transition magnitude

Reason:

- factored channels preserve edit locality
- monolithic joint tokens would collapse back toward opaque learned tokens

## What the representation must make easy

### 1. Local symbolic edits

Example:

- change `right_hand_motion = opening` while preserving unrelated fields

Why:

- this is where symbolic structure already beats proxies

### 2. Transition-conditioned retrieval

Example:

- retrieve donor motifs that match a target previous-state and realized motion

Why:

- current local evidence says current-frame donors are structurally insufficient

### 3. Family-level repair

Example:

- post-hoc reconcile near-confusions within anatomically related groups

Why:

- already a verified strength and should stay part of the system view

### 4. Interaction-aware control

Example:

- edit hand motion without unintentionally changing cross-hand relation

Why:

- this is the key unresolved frontier

## What the representation must not depend on

Do not make the mainline depend on:

- manual semantic gesture categories
- full-body linguistic notation units
- opaque learned codebooks
- a specific temporal backbone

## Evaluation implications by channel

### State channels

Need:

- backward compatibility checks with frame-wise HL
- grouped-vs-flat representation ablation

### Transition channels

Need:

- transition-conditioned motif retrieval
- donor-only oracle failure as motivation
- realized edit or reconstruction under previous-frame conditioning

### Coordination channels

Need:

- interaction-vs-noninteraction slice analysis
- collateral drift under interacting-hand edits

## Immediate experiment priorities implied by this design

Priority 1:

- build a transition-conditioned symbolic editor using the mainline channels

Priority 2:

- add a focused interaction-aware edit benchmark

Priority 3:

- test whether cross-hand coordination channels reduce the interaction weakness

Priority 4:

- keep persistence and phase as auxiliary ablations, not the mainline identity

## Decision rule for future additions

A new symbolic channel should be promoted to mainline only if it satisfies all
three:

1. It is explicit in the representation rather than hidden in the encoder.
2. It improves a task where interpretability or controllability matters.
3. Its benefit cannot be replaced by "old HL + stronger temporal model".
