# Known limitations to resolve or state

- The local source is Mendeley V1 (dataset version 1.0, DOI `10.17632/x493z8nmwc.1`), while `model_generation: v2` is an internal project label. Audit counts are 1,551 source records and 1,545 analysis records after six documented exclusions.
- Infant/session/family identifiers may be absent or incomplete. Exact-content/source grouping does not establish subject-independent evaluation.
- InfantCry-DBL tiers may differ in recording conditions and provenance. In-tier performance does not establish cross-tier or real-world generalization.
- The Mendeley V1 record recommends 16 kHz mono for uniform inputs; both Stage 2 feature profiles are locked to 16 kHz mono.
- Tier 2 `clip_id` is used for grouped splitting as recommended by the Dataset record, but its real-world subject/session semantics have not been independently verified. Do not claim subject-independent evaluation.
- DBL labels describe dataset annotation categories and are not clinically verified diagnoses.
- A Stage 2-only study receives known DBL clips; it does not assess rejection of non-infant audio or the full two-stage cascade.
- A paper reproduction cannot be called exact until the source data version, labels, splits, input modalities, preprocessing, training procedure, and evaluation are verified.
- The MMT + ResLSTM candidate is informed by Qiu et al. (2024), but the local implementation uses TensorFlow/Keras, the locally audited release, a grouped train/validation/locked-test split, and a shared project training schedule rather than the paper's full recipe. It is therefore not an exact reproduction.
- The paper reports an MMT output geometry but does not unambiguously document the axis of its first feature concatenation. The implementation's `(84,48,1)` fusion order is a geometry-consistent interpretation; results should not be described as using a verified identical tensor layout.
- Candidate pipelines use different feature representations. A validation/test difference compares the complete registered pipelines and does not isolate the causal contribution of the architecture from the features. Feature/model ablations are needed for component-level claims.
- MMT feature details are a documented local implementation interpretation of the paper; the sample-rate profiles are locked to 16 kHz, while the paper's exact fusion-axis interpretation remains uncertain.
- The one-shot test batch produces paired descriptive metrics for every preregistered candidate after validation selection. Test metrics must not be used for tuning, re-ranking, model selection, or promotion; after a claim, the same split cannot be evaluated again.
- A single public corpus and internal test do not constitute independent external validation.
- High accuracy alone is not evidence of broad generalization or Scopus acceptance.
- Tier-transfer execution, repeated-seed aggregation, ablations/augmentation, and a Sinc-Mamba-KAN comparator are not implemented or run in this scaffold; they need separate pre-specified protocols.
- The locked split is `seed-42-a241c8eba2da`; the independent group identity remains unverified. The locked-test access guard prevents accidental repeat evaluation in the provided CLI, but cannot stop a user from copying raw files or editing local evidence manually; protect the split and report every protocol change.
