# Stage 2 research protocol (draft)

## Research question

How do the selected audio representation and classifier family affect five-class InfantCry-DBL cry-type classification when candidate runs use the same audited records, grouped Train/Validation split, class order, random seed, and training controls?

This is a paper-informed local comparison, not an exact reproduction of Qiu et al. (2024), and not a clinical diagnostic study.

## Dataset audit and analysis set

Use one verified InfantCry-DBL release: dataset version `1.0` (Mendeley V1, DOI `10.17632/x493z8nmwc.1`). `model_generation: v2` is an internal project/model label, not a Dataset release number. Keep the source labels unchanged. The audit reports 1,551 source records; six records from three exact-content families with conflicting labels are documented in the exclusion ledger and excluded from analysis, leaving 1,545. Do not claim infant-independent evaluation without a verified infant identifier.

The primary Tier 2 split is stratified and group-aware, 70/15/15, using the Dataset-recommended `clip_id` grouping field and seed 42. Its frozen ID is `seed-42-a241c8eba2da` (Train 848, Validation 183, Test 180). The identity semantics of `clip_id` have not been independently verified, so this prevents overlap only at that identifier level; it does not establish subject-independent generalization.

Audit and split construction inspect eligible audio and label metadata before the split is frozen: the audit checks file integrity and stratification uses labels to construct a feasible partition. After freezing, candidate training reads the hash-verified `train_validation_records.csv` and Train/Validation audio only. Test rows/audio are not used for fitting, checkpoint selection, candidate selection, or threshold tuning.

The prior locked-Test attempt for this split was claimed and failed. It must not be retried, reset, or deleted. The present implementation work is limited to synthetic tests; it does not train on the real corpus or access Test records. Any future confirmatory test estimate requires an advisor-approved protocol and a fresh untouched test partition or independent dataset.

Tier 1 (`Dunstan-Core`) transfer experiments are separate analyses. Cross-tier results must not select models or tune thresholds for the primary Tier 2 comparison.

## Reference paper and scope of alignment

The reference is Qiu et al., “Classification of Infant Cry Based on Hybrid Audio Features and ResLSTM,” *Journal of Voice* (2024), DOI [10.1016/j.jvoice.2024.08.022](https://doi.org/10.1016/j.jvoice.2024.08.022). The source PDF is retained at [`research/Qiu_2024_Hybrid_Audio_Features_ResLSTM.pdf`](../research/Qiu_2024_Hybrid_Audio_Features_ResLSTM.pdf).

| Topic | Reported in the article | Local implementation / limitation |
|---|---|---|
| Main representation | MMT: MFCC, Mel spectrogram, and Tonnetz | `mmt` feature profile, with a documented local fusion interpretation below |
| DBL feature geometry | `T=24`; MFCC `(42,T)` doubled along time; Tonnetz `(6,T)` expanded to `(42,T)`; Mel `(42,T)`; final `(84,2T)` | MMT tensor `(84,48,1)`; layout is internally consistent with the written shape but is not confirmed as the authors' exact concatenation order |
| Main model | Residual CNN + stacked unidirectional LSTM (ResLSTM) | New `qiu_mmt_reslstm` uses explicit padding; historical `mmt_reslstm` remains unchanged and uses the previous Keras `same`-padding implementation |
| Reported framework and recipe | PyTorch 2.0.1; random 10-fold CV; batch size 128; learning rate `5e-4`; up to 2,000 epochs | TensorFlow/Keras; locally audited DBL V1; frozen grouped Train/Validation split; shared project schedule (50 epochs maximum, batch 32, Adam `0.001`, early-stopping patience 8). These differences preclude describing this as an exact reproduction. |
| Unspecified local design choices | The article does not settle every input-fusion and comparator-head detail needed to reproduce its complete pipeline | These choices are identified as project decisions, not attributed to the authors. |

For the local MMT interpretation, Tonnetz has six bands; repeat each band seven times to obtain `(42,T)`, concatenate Tonnetz then Mel along time into `(42,2T)`, then concatenate that block with time-doubled MFCC along the feature axis. This produces `(84,48,1)` for `T=24`. The paper's written description does not unambiguously identify the first concatenation axis; confirmation from author code or supplementary material is still needed before claiming exact tensor-layout reproduction.

## Dedicated Qiu-informed comparison

The separate config [`configs/models_stage2_qiu_comparison.yaml`](../configs/models_stage2_qiu_comparison.yaml) enables exactly four candidates. It does not alter [`configs/models_stage2.yaml`](../configs/models_stage2.yaml), its candidate set, historical run folders, or the existing Webapp bundle.

| Candidate ID | Input | Classifier | Purpose and local choices |
|---|---|---|---|
| `qiu_mfcc_reslstm` | MFCC-only `(42,24,1)` | Explicitly padded residual CNN + unidirectional LSTM `[256,128,64]` | MFCC-only comparator with the ResLSTM family; the fixed MFCC frame contract is a local choice |
| `qiu_mmt_resnet` | MMT `(84,48,1)` | Explicitly padded residual CNN + global-average-pooling classification head | Tests the residual CNN without recurrence; global pooling/head details are project choices |
| `qiu_mmt_lstm` | MMT `(84,48,1)` | Flatten frequency/channel bins per time frame, then unidirectional LSTM `[256,128,64]` | Tests recurrence without the residual CNN; sequence conversion is a project choice |
| `qiu_mmt_reslstm` | MMT `(84,48,1)` | Explicitly padded residual CNN + unidirectional LSTM `[256,128,64]` | Closest local structural comparator to the paper's reported MMT + ResLSTM method |

For the residual candidates, the local Qiu-aligned builder uses a 7×7, 32-filter stem preceded by two-pixel zero padding, followed by residual stages `[64,128,256,512]`. Each 3×3 residual convolution is preceded by one-pixel zero padding; skip-path projections use 1×1 convolutions. ResLSTM variants use dropout `0.3` in residual blocks and five-class softmax output. The MMT ResNet's global-pooling head and the MMT LSTM-only candidate are project-defined comparators, not claims about additional article architectures.

The MFCC-only profile uses 42 MFCC coefficients and fixes the time axis at 24 frames to produce `(42,24,1)`. This is a comparison condition and does not mean the article's primary MMT method is MFCC-only. The MMT profile uses the documented local `(84,48,1)` interpretation. All profiles use 16 kHz mono preprocessing in this project; it is a project decision consistent with the dataset recommendation, not a claim that Qiu et al. used the identical preprocessing.

All four candidates in the dedicated config declare the same project training schedule and production runtime: at most 50 epochs, batch size 32, Adam with learning rate `0.001`, early-stopping patience 8, deterministic operations, validation macro-F1 checkpoint/selection, GPU, and mixed precision. They use the same frozen Train/Validation IDs and seed. Candidate-specific feature profiles are intentionally different and frozen in the run evidence. Thus the comparison evaluates registered pipelines; a difference in score cannot be attributed to architecture alone when feature representation also differs.

The existing main model and the four Qiu-informed conditions are distinct comparisons and must not be silently combined into one selection set. A study comparing the main pipeline against the Qiu set should preregister a separate config with a deliberate candidate list and matching shared controls.

## Existing project candidates and audio features

The main candidate set remains defined by `configs/models_stage2.yaml`: Majority, MFCC-SVM, Log-Mel CNN, CNN-BiLSTM-Temporal-Attention, and the historical MMT + ResLSTM candidate. Its `stage2_hybrid` profile uses 16,000 Hz mono audio, silence trimming at `top_db=20`, peak normalization, MFCC 40, Delta 40, Delta2 40, Log-Mel 64 and Chroma 12, padded/truncated to 128 frames (`(196,128,1)`). The separate Qiu config does not change this profile.

Feature scaling or any learned preprocessing must be fitted on Train only. Deterministic audio extraction itself uses the per-candidate frozen feature config; no whole-dataset scaler is fit during extraction.

## Validation selection and reporting

Validation macro-F1 is the predeclared primary selection metric. Each candidate is fitted on Train and uses Validation for checkpoint selection. `select-candidate` requires exactly one completed run for every enabled candidate in the same config, with the same split, dataset evidence, class order, seed, and shared training controls. Declared feature/model conditions may differ. The command writes one immutable selection record; score ties are resolved by candidate order in that frozen config.

No Test evaluation is part of this implementation. The existing split's locked-Test event is not reusable because its prior claim failed. If an approved future test protocol uses a fresh untouched partition, it must evaluate the complete frozen candidate set once after validation selection, use identical IDs for every candidate, and keep all test results out of selection, re-ranking, threshold tuning, and promotion decisions.

For an authorized future evaluation, report accuracy, balanced accuracy, per-class precision/recall/F1 and support, plus uncertainty intervals only when verified grouping supports them. Do not present uncalibrated class scores as clinical probabilities. Do not combine published paper scores with local metrics in one ranking; paper numbers are context unless a separately verified reproduction protocol is run.

## Reproducibility and status

Each candidate run freezes dataset and manifest hashes, split IDs/hashes, class order, feature/model config, seed, environment, runtime/device, validation metrics, and artifact hashes. No results are included in this protocol. The Qiu comparison config and model paths have synthetic/unit-test coverage only; no real-data training or end-to-end GPU training has been performed as part of this alignment work.
