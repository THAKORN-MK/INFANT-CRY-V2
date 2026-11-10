"""Feature loading for an explicitly supplied training/validation partition."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from infantcry_research.audio.features import (
    AudioConfig,
    extract_audio_features,
    load_and_preprocess_audio,
)
from infantcry_research.data.manifest import Record


def load_feature_partition(
    records: Mapping[str, Record], sample_ids: Sequence[str], config: AudioConfig
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Load and extract features for only the IDs explicitly provided by caller.

    This function is deliberately unaware of other split partitions. Passing
    only train IDs and validation IDs prevents locked-test waveforms from being
    materialized during training.
    """

    selected_ids = tuple(sample_ids)
    if not selected_ids:
        raise ValueError("sample_ids must contain at least one record")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("sample_ids contains duplicates")
    unknown = sorted(set(selected_ids) - set(records))
    if unknown:
        raise ValueError(f"unknown sample IDs: {unknown[:10]}")

    features: list[np.ndarray] = []
    labels: list[str] = []
    for sample_id in selected_ids:
        record = records[sample_id]
        waveform, sample_rate = load_and_preprocess_audio(record.path, config)
        tensor = extract_audio_features(waveform, sample_rate, config)
        features.append(tensor)
        labels.append(record.raw_label)
    return np.stack(features, axis=0), tuple(labels), selected_ids
