# InfantCry-DBL data provenance

## Primary dataset

This workspace uses the **InfantCry-DBL V1 release** (dataset version `1.0`, DOI `10.17632/x493z8nmwc.1`) as its source dataset. The internal `model_generation: v2` and the local folder suffix `_v2` identify this project's model iteration; they are not the Dataset release version. The official V1 record describes 1,551 WAV clips across `Dunstan-Core` (337 studio clips) and `InfantCry-1214` (1,214 in-the-wild clips), with original DBL labels `eairh`, `eh`, `heh`, `neh`, and `owh`, and CC BY 4.0. It identifies the parent folder as authoritative for class labels and describes a per-clip `metadata.csv`; it does not establish that every local copy contains verified subject/session identifiers. Review both the public record and the bundled README/DATASET_DESCRIPTION before using or redistributing any data:

- Dataset record: <https://data.mendeley.com/datasets/x493z8nmwc/1>
- Cited Sinc-Mamba-KAN article: <https://doi.org/10.1109/ACCESS.2026.3733522>

The local copy in `data/raw/infantcry_dbl_v2/` has been audited as dataset release `1.0`. The audit found 1,551 metadata records; six records in three exact-content families with conflicting labels are excluded from analysis by a SHA-256-bound curation ledger, leaving 1,545 records. The raw audio and original metadata were not edited. The corrected audit and locked split are recorded under `manifests/model_v2_dataset_v1/` and `manifests/splits/model_v2_dataset_v1/`. These paths make the model generation and Dataset release explicit.

## Local data handling

- The user supplies and copies the audio; this project does not download, rename, or move it.
- Keep the original directory structure and metadata. Generated manifests should store relative paths, labels as supplied, tier, available group IDs, and file SHA-256.
- Record exclusions and reasons. Never silently remove unreadable files, duplicates, or unknown labels.
- Raw data, derived features, and model/run artifacts are local-only and ignored by Git.
- Cite the dataset and retain attribution consistent with its published license. Confirm the current license and terms before external redistribution.

## Label interpretation

Model targets remain the dataset's original five DBL labels. Any mapping to descriptive categories such as hunger or discomfort must be supported by the dataset's documentation and stored separately from the original label. These labels are not clinical diagnoses.

## Version control

The main benchmark uses one release only: Mendeley V1 (`10.17632/x493z8nmwc.1`). If a later project chooses another release, it will require a separate audit, split, and protocol; records from different releases must not be silently mixed.

The public record describes multiple Tier-2 source streams and volunteer family recordings, but a usable, verified infant-ID field is not guaranteed in the per-clip manifest. `grouping_identity_verified` therefore defaults to `false`. Enable it only after checking the release documentation and confirming that the configured field identifies the independent unit used for the intended uncertainty analysis. If such identifiers are absent, report the strongest available grouping level, omit group-level confidence intervals, and do not claim subject-independent evaluation. When both metadata and parent-folder labels exist, enable `cross_check_parent_label` only after confirming the configured folder depth; conflicting labels must be resolved from source evidence before splitting.
