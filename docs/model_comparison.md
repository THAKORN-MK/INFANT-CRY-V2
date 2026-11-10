# Model comparison plan

## Existing Stage 2 candidate set

The primary project candidate set remains in [`configs/models_stage2.yaml`](../configs/models_stage2.yaml). It is unchanged by the paper-informed comparison.

| Candidate | Input | Role |
|---|---|---|
| Majority | Training labels only | Non-audio floor baseline |
| MFCC-SVM | Per-clip MFCC mean/std | Classical acoustic baseline; scaling fitted on Train only |
| Log-Mel CNN | 64-bin Log-Mel slice | Compact neural baseline |
| CNN-BiLSTM-Attention | MFCC + Delta + Delta2 + Log-Mel + Chroma `(196,128,1)` | Existing project neural candidate |
| MMT + ResLSTM (`mmt_reslstm`) | MFCC + Mel + expanded Tonnetz `(84,48,1)` | Existing paper-informed candidate; its historical implementation and runs are preserved |

The CNN-BiLSTM-Attention model uses the existing implementation in `src/infantcry_research/models/cnn_bilstm_attention.py`. The Qiu alignment work does not change it. Sinc-Mamba-KAN remains disabled and is outside this comparison because its complete method and input protocol have not been verified.

## Separate Qiu-informed comparison set

The dedicated [`configs/models_stage2_qiu_comparison.yaml`](../configs/models_stage2_qiu_comparison.yaml) enables exactly the four candidates below. Use this config as its own preregistered comparison set. Do not mix its runs with the main candidate set or change `configs/models_stage2.yaml` implicitly.

| Candidate ID | Feature tensor | Model path | What the comparison probes |
|---|---|---|---|
| `qiu_mfcc_reslstm` | MFCC-only `(42,24,1)` | Explicitly padded residual CNN → stacked unidirectional LSTM `[256,128,64]` → 5-class softmax | ResLSTM family with MFCC-only input |
| `qiu_mmt_resnet` | MMT `(84,48,1)` | Explicitly padded residual CNN → global average pooling → 5-class softmax | Residual CNN without recurrent layers |
| `qiu_mmt_lstm` | MMT `(84,48,1)` | Frequency/channel vector per time frame → unidirectional LSTM `[256,128,64]` → 5-class softmax | Recurrent classifier without residual CNN |
| `qiu_mmt_reslstm` | MMT `(84,48,1)` | Explicitly padded residual CNN → stacked unidirectional LSTM `[256,128,64]` → 5-class softmax | Closest local structural comparator to Qiu et al.'s reported main method |

The article reports the MMT features and the ResLSTM combination. The MFCC-only, MMT-ResNet, and MMT-LSTM rows are controlled project comparators to separate parts of that pipeline; they must not be attributed to the authors unless verified in the article's complete experiments. The global-pooling head, frame-wise frequency flattening, and fixed MFCC-only geometry are local design decisions.

### Shared comparison controls

All four candidates are frozen in the same dedicated config and use the same audited dataset records, Train/Validation split IDs, class order, and seed. The shared training controls are maximum 50 epochs, batch size 32, Adam with learning rate `0.001`, early-stopping patience 8, deterministic operations, and validation macro-F1 for checkpoint selection. The production config declares GPU and mixed precision for all four. Candidate-specific feature profiles are expected: MFCC-only uses `(42,24,1)` and MMT uses `(84,48,1)`.

For the two ResLSTM candidates and the MMT ResNet, explicit zero padding precedes the 7×7 stem convolution (two pixels per side) and each 3×3 residual convolution (one pixel per side). The residual stages use `[64,128,256,512]` channels; the stem has 32 filters. ResLSTM candidates use dropout `0.3` in the residual blocks. The MMT ResNet uses a project-chosen global-average-pooling head. The MMT LSTM uses a project-chosen sequence conversion and no convolutional layers.

These are comparisons of registered full pipelines. The MFCC-only ResLSTM versus MMT ResLSTM changes the feature representation as well as tensor geometry; differences cannot be interpreted as a pure architecture effect. The MMT ResNet/LSTM/ResLSTM conditions share the MMT profile and are more suitable for probing classifier-family differences, while still retaining other model-design differences.

## Selection and test policy

Validation macro-F1 ranks candidates only after one complete validation-only run exists for every enabled ID in the same config. The selector checks dataset evidence, split hash, seed, class order, and shared training controls; it allows the preregistered feature/model differences. Ties follow config order. No test score may select, re-rank, tune, or promote a candidate.

The prior locked-Test attempt for split `seed-42-a241c8eba2da` was claimed and failed. Do not retry, reset, or delete that ledger. This implementation does not evaluate Test. A future confirmatory comparison needs advisor approval and a fresh untouched test partition or independent dataset, followed by one paired evaluation of the full frozen candidate set.

No performance results are reported in this document. The new builders, dispatch, candidate selection, and no-Test-access behavior have synthetic/unit-test coverage only; no real-corpus training or GPU training was performed during implementation.

For article scope, local feature interpretation, and reproduction limits, see the [research protocol](research_protocol.md).
