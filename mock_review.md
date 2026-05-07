# Mock Review Summary

## Overall verdict

If the paper is positioned as:

- a **representation study**,
- with the main contribution being **explicit motion and keyframe semantics**,
- and with the main empirical claim being **comparable recoverability under fair training rather than universal reconstruction gains**,

then there is currently **no obvious fatal flaw** in the evidence chain.

## What would have been fatal, but is now handled

### 1. "The result is a lucky run."

Handled by:
- multi-seed analysis,
- warm-start stabilization,
- fair continued-training control.

Conclusion:
- the original scratch advantage is not robust,
- but the final paper no longer depends on that overclaim.

### 2. "The gain comes from extra training budget."

Handled by:
- continued-training control for Static-HL.

Conclusion:
- yes, extra training budget explains the apparent warm-start reconstruction advantage,
- so the paper must not claim lower reconstruction error as its core contribution.
- this has now been reflected in the draft.

### 3. "The split is suspicious."

Handled by:
- explicit documentation that `source_split` is provenance metadata only,
- the paper benchmark is the reorganized clip-level split.

Conclusion:
- still worth explaining carefully,
- but not fatal if written clearly.

## Remaining weaknesses that are real but manageable

### W1. The strongest practical gain is semantic richness, not reconstruction error.

Implication:
- venue fit matters.
- this is more convincing as a symbolic representation / documentation paper than as a pure performance paper.

### W2. The auxiliary downstream tasks are weak.

Implication:
- do not oversell classification or retrieval.
- use them only as supporting analysis.

### W3. Keyframe detection is heuristic.

Implication:
- acceptable as a limitation/future work item,
- not fatal for the current story.

## Suggested final paper stance

The paper should say:

1. Framewise HL lacks explicit temporal semantics.
2. Temporal-HL adds motion and keyframe channels.
3. These channels are learnable and interpretable.
4. Under fair training, Temporal-HL preserves motion recoverability at a level comparable to Static-HL.
5. Therefore Temporal-HL offers a richer symbolic language without sacrificing downstream usability.

The paper should avoid saying:

1. Temporal-HL universally improves reconstruction.
2. Temporal-HL is a stronger representation under all decoders.
3. The main novelty is lower numerical error.

## Reviewer confidence estimate

With the current evidence and honest framing:

- fatal issue risk: low
- strong-reject risk due to overclaiming: low, if the draft is updated accordingly
- moderate-reject risk due to venue fit / limited downstream tasks: medium
- accept / borderline accept potential for a suitable venue: reasonable
