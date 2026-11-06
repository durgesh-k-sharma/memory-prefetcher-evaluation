# Memory Prefetcher Characterization

This context describes how memory prefetchers are evaluated and compared across workloads, cache levels, and memory-access patterns.

## Deliverables

**Midsem checkpoint**:
The intermediate report of the four required L2 prefetch configurations across the ten named workloads, including performance metrics and memory-access characterization.
_Avoid_: Midterm report

**Endsem evaluation**:
The final report that adds workload-level root-cause analysis, miss attribution, the IP-stride improvement, Pythia at L2, and Berti in a separate L1D scenario.
_Avoid_: Final comparison (when referring to the full deliverable)

## Evaluation

**Same-level comparison**:
A comparison among prefetch configurations operating at the same cache level and measured against a matching no-prefetch baseline.
_Avoid_: Cross-level ranking

**Memory-access characterization**:
Analysis of a workload's PC-local address deltas, ordered delta sequences, and recurring cache-line footprints within pages.
_Avoid_: Access-pattern guess

**Miss attribution**:
Classification of a demand miss by why a prefetcher did not prevent it, using evidence available from that prefetcher's mechanism.
_Avoid_: Assigning a universal cause label

**Feedback-directed IP-stride variant**:
An IP-stride prefetcher that keeps its stride prediction and adjusts a control decision in response to measured prefetch behavior.
_Avoid_: Replacement prefetcher

**Prefetch address episode**:
A tracked cache-line address from an accepted prefetch submission until a demand use, eviction, or demand fill resolves its usefulness outcome. Duplicate submissions to the same pending address remain part of one episode.
