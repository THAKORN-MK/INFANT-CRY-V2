"""Deterministic audio normalization and Stage 2 feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np


FEATURE_ORDER = ("mfcc", "delta", "delta2", "logmel", "chroma")
MMT_COMPONENT_ORDER = ("mfcc_time_doubled", "tonnetz_then_mel_time_concat")
SUPPORTED_FEATURE_SETS = ("stage2_hybrid", "mmt", "mfcc_paper")


class AudioContractError(ValueError):
    """Raised when an audio clip violates the configured signal contract."""


@dataclass(frozen=True)
class AudioConfig:
    feature_set: str = "stage2_hybrid"
    sample_rate: int = 22_050
    mono: bool = True
    trim_top_db: float = 20.0
    normalize_waveform: bool = True
    n_mfcc: int = 40
    n_mels: int = 64
    n_chroma: int = 12
    n_fft: int = 2_048
    hop_length: int = 512
    delta_width: int = 9
    max_frames: int = 128
    mmt_frames: int = 24
    mfcc_frames: int = 24
    feature_order: tuple[str, ...] = FEATURE_ORDER
    mmt_component_order: tuple[str, ...] = MMT_COMPONENT_ORDER

    @property
    def feature_bins(self) -> int:
        if self.feature_set == "mmt":
            return 84
        if self.feature_set == "mfcc_paper":
            return self.n_mfcc
        return 3 * self.n_mfcc + self.n_mels + self.n_chroma

    @property
    def tensor_shape(self) -> tuple[int, int, int]:
        """Return the fixed model-input shape for the selected feature set."""

        if self.feature_set == "mmt":
            time_bins = 2 * self.mmt_frames
        elif self.feature_set == "mfcc_paper":
            time_bins = self.mfcc_frames
        else:
            time_bins = self.max_frames
        return (self.feature_bins, time_bins, 1)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> AudioConfig:
        """Build a checked config from the `features` section of a YAML file."""

        config = cls(
            feature_set=str(payload.get("feature_set", cls.feature_set)),
            sample_rate=int(payload.get("sample_rate", cls.sample_rate)),
            mono=bool(payload.get("mono", cls.mono)),
            trim_top_db=float(payload.get("trim_top_db", cls.trim_top_db)),
            normalize_waveform=bool(payload.get("normalize_waveform", cls.normalize_waveform)),
            n_mfcc=int(payload.get("n_mfcc", cls.n_mfcc)),
            n_mels=int(payload.get("n_mels", cls.n_mels)),
            n_chroma=int(payload.get("n_chroma", cls.n_chroma)),
            n_fft=int(payload.get("n_fft", cls.n_fft)),
            hop_length=int(payload.get("hop_length", cls.hop_length)),
            delta_width=int(payload.get("delta_width", cls.delta_width)),
            max_frames=int(payload.get("max_frames", cls.max_frames)),
            mmt_frames=int(payload.get("mmt_frames", cls.mmt_frames)),
            mfcc_frames=int(payload.get("mfcc_frames", cls.mfcc_frames)),
            feature_order=tuple(payload.get("feature_order", FEATURE_ORDER)),
            mmt_component_order=tuple(
                payload.get("mmt_component_order", MMT_COMPONENT_ORDER)
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.feature_set not in SUPPORTED_FEATURE_SETS:
            raise AudioContractError(
                f"feature_set must be one of {SUPPORTED_FEATURE_SETS}, got {self.feature_set!r}"
            )
        if self.sample_rate <= 0 or self.n_fft <= 0 or self.hop_length <= 0:
            raise AudioContractError("sample_rate, n_fft, and hop_length must be positive")
        if not self.mono:
            raise AudioContractError("The current Stage 2 tensor contract requires mono audio")
        if self.n_mfcc <= 0 or self.n_mels < self.n_mfcc or self.n_chroma <= 0:
            raise AudioContractError("Require positive feature counts and n_mels >= n_mfcc")
        if self.delta_width < 3 or self.delta_width % 2 == 0:
            raise AudioContractError("delta_width must be an odd integer >= 3")
        if self.max_frames <= 0:
            raise AudioContractError("max_frames must be positive")
        if not np.isfinite(self.trim_top_db) or self.trim_top_db <= 0:
            raise AudioContractError("trim_top_db must be finite and positive")
        if self.feature_order != FEATURE_ORDER:
            if self.feature_set == "stage2_hybrid":
                raise AudioContractError(f"feature_order must be exactly {FEATURE_ORDER}")
        if self.feature_set == "mmt":
            if self.n_mfcc != 42 or self.n_mels != 42:
                raise AudioContractError("MMT requires n_mfcc=42 and n_mels=42")
            if self.mmt_frames <= 0:
                raise AudioContractError("mmt_frames must be positive")
            if self.mmt_component_order != MMT_COMPONENT_ORDER:
                raise AudioContractError(
                    f"mmt_component_order must be exactly {MMT_COMPONENT_ORDER}"
                )
        if self.feature_set == "mfcc_paper":
            if self.n_mfcc != 42 or self.n_mels != 42:
                raise AudioContractError(
                    "mfcc_paper requires n_mfcc=42 and n_mels=42"
                )
            if self.mfcc_frames <= 0:
                raise AudioContractError("mfcc_frames must be positive")


def load_and_preprocess_audio(path: Path, config: AudioConfig) -> tuple[np.ndarray, int]:
    """Read an audio file, downmix/resample, trim silence, and peak-normalize it."""

    config.validate()
    file_path = Path(path)
    if not file_path.is_file():
        raise AudioContractError(f"Audio file does not exist: {file_path}")
    try:
        import soundfile as sf

        waveform, source_rate = sf.read(file_path, dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioContractError(f"Could not decode audio file {file_path}: {exc}") from exc
    return preprocess_waveform(waveform, int(source_rate), config), config.sample_rate


def preprocess_waveform(
    waveform: np.ndarray, sample_rate: int, config: AudioConfig
) -> np.ndarray:
    """Return a finite, mono, trimmed, peak-normalized float32 waveform."""

    config.validate()
    if sample_rate <= 0:
        raise AudioContractError("sample rate must be positive")
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 2:
        if audio.shape[1] <= 8:
            audio = np.mean(audio, axis=1, dtype=np.float32)
        elif audio.shape[0] <= 8:
            audio = np.mean(audio, axis=0, dtype=np.float32)
        else:
            raise AudioContractError(f"Could not identify audio channels in shape {audio.shape}")
    if audio.ndim != 1 or audio.size == 0:
        raise AudioContractError("waveform must be a non-empty mono array or a samples-by-channels matrix")
    if not np.isfinite(audio).all():
        raise AudioContractError("waveform contains NaN or infinite samples")
    if sample_rate != config.sample_rate:
        audio = librosa.resample(
            audio, orig_sr=sample_rate, target_sr=config.sample_rate
        ).astype(np.float32, copy=False)
    if audio.size == 0:
        raise AudioContractError("waveform became empty during resampling")

    trimmed, _ = librosa.effects.trim(audio, top_db=config.trim_top_db)
    if trimmed.size == 0:
        raise AudioContractError("audio is silent after silence trimming")
    peak = float(np.max(np.abs(trimmed)))
    if not np.isfinite(peak) or peak <= np.finfo(np.float32).eps:
        raise AudioContractError("audio is silent after silence trimming")
    if config.normalize_waveform:
        trimmed = trimmed / peak
    return np.asarray(trimmed, dtype=np.float32)


def extract_stage2_features(
    waveform: np.ndarray, sample_rate: int, config: AudioConfig
) -> np.ndarray:
    """Create `(feature_bins, max_frames, 1)` MFCC/Mel/Chroma tensor.

    Frames shorter than the Delta window are edge-padded to three frames before
    derivative computation. The fixed-width representation is then right-zero-
    padded or truncated from the end of the sequence. No dataset-level scaler is
    fitted here, so there is no train/test statistics leakage.
    """

    config.validate()
    if config.feature_set != "stage2_hybrid":
        raise AudioContractError(
            "extract_stage2_features only supports stage2_hybrid; use extract_audio_features"
        )
    if sample_rate != config.sample_rate:
        raise AudioContractError(
            f"feature extraction expected sample rate {config.sample_rate}, got {sample_rate}"
        )
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise AudioContractError("feature extraction expects a non-empty mono waveform")
    if not np.isfinite(audio).all():
        raise AudioContractError("waveform contains NaN or infinite samples")
    if float(np.max(np.abs(audio))) <= np.finfo(np.float32).eps:
        raise AudioContractError("cannot extract features from silent audio")

    feature_audio = audio
    if feature_audio.size < config.n_fft:
        feature_audio = np.pad(
            feature_audio, (0, config.n_fft - feature_audio.size), mode="constant"
        )

    mel_power = librosa.feature.melspectrogram(
        y=feature_audio,
        sr=sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel_power, ref=np.max).astype(np.float32, copy=False)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=config.n_mfcc).astype(np.float32, copy=False)
    mfcc_for_delta = _ensure_delta_frames(mfcc)
    delta_width = min(config.delta_width, _largest_odd_at_most(mfcc_for_delta.shape[1]))
    delta = librosa.feature.delta(
        mfcc_for_delta, width=delta_width, order=1, mode="nearest"
    ).astype(np.float32, copy=False)
    delta2 = librosa.feature.delta(
        mfcc_for_delta, width=delta_width, order=2, mode="nearest"
    ).astype(np.float32, copy=False)
    chroma = librosa.feature.chroma_stft(
        y=feature_audio,
        sr=sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_chroma=config.n_chroma,
    ).astype(np.float32, copy=False)

    frame_count = max(mfcc_for_delta.shape[1], logmel.shape[1], chroma.shape[1])
    mfcc_for_delta = _fit_minimum_frames(mfcc_for_delta, frame_count)
    logmel = _fit_minimum_frames(logmel, frame_count)
    chroma = _fit_minimum_frames(chroma, frame_count)
    delta = _fit_minimum_frames(delta, frame_count)
    delta2 = _fit_minimum_frames(delta2, frame_count)
    features = np.concatenate((mfcc_for_delta, delta, delta2, logmel, chroma), axis=0)
    features = _fit_time_axis(features, config.max_frames)
    if features.shape != (config.feature_bins, config.max_frames):
        raise AudioContractError(
            f"internal feature shape mismatch: expected {(config.feature_bins, config.max_frames)}, "
            f"got {features.shape}"
        )
    tensor = features[:, :, np.newaxis]
    if not np.isfinite(tensor).all():
        raise AudioContractError("feature extraction produced NaN or infinite values")
    return np.asarray(tensor, dtype=np.float32)


def extract_audio_features(
    waveform: np.ndarray, sample_rate: int, config: AudioConfig
) -> np.ndarray:
    """Dispatch to the fixed tensor contract selected by ``config.feature_set``."""

    config.validate()
    if config.feature_set == "stage2_hybrid":
        return extract_stage2_features(waveform, sample_rate, config)
    if config.feature_set == "mfcc_paper":
        return extract_mfcc_paper_features(waveform, sample_rate, config)
    return extract_mmt_features(waveform, sample_rate, config)


def extract_mfcc_paper_features(
    waveform: np.ndarray, sample_rate: int, config: AudioConfig
) -> np.ndarray:
    """Build the paper-profile MFCC-only `(42, mfcc_frames, 1)` tensor.

    This uses the same 42-coefficient MFCC component as the local MMT
    extractor, but does not append derivatives, Mel, Tonnetz, or other
    features. Time is zero-padded or truncated to the configured frame count.
    """

    config.validate()
    if config.feature_set != "mfcc_paper":
        raise AudioContractError(
            "extract_mfcc_paper_features requires feature_set='mfcc_paper'"
        )
    if sample_rate != config.sample_rate:
        raise AudioContractError(
            f"feature extraction expected sample rate {config.sample_rate}, got {sample_rate}"
        )
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise AudioContractError("feature extraction expects a non-empty mono waveform")
    if not np.isfinite(audio).all():
        raise AudioContractError("waveform contains NaN or infinite samples")
    if float(np.max(np.abs(audio))) <= np.finfo(np.float32).eps:
        raise AudioContractError("cannot extract features from silent audio")

    feature_audio = audio
    if feature_audio.size < config.n_fft:
        feature_audio = np.pad(
            feature_audio, (0, config.n_fft - feature_audio.size), mode="constant"
        )

    try:
        mel_power = librosa.feature.melspectrogram(
            y=feature_audio,
            sr=sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            power=2.0,
        ).astype(np.float32, copy=False)
        log_mel = librosa.power_to_db(mel_power, ref=np.max).astype(
            np.float32, copy=False
        )
        mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=config.n_mfcc).astype(
            np.float32, copy=False
        )
    except Exception as exc:
        raise AudioContractError(f"could not extract MFCC features: {exc}") from exc

    if mfcc.ndim != 2 or mfcc.shape[0] != 42:
        raise AudioContractError(f"MFCC must have shape (42, frames), got {mfcc.shape}")
    features = _fit_time_axis(mfcc, config.mfcc_frames)
    if features.shape != config.tensor_shape[:2]:
        raise AudioContractError(
            f"internal MFCC shape mismatch: expected {config.tensor_shape[:2]}, got {features.shape}"
        )
    tensor = features[:, :, np.newaxis]
    if not np.isfinite(tensor).all():
        raise AudioContractError("MFCC extraction produced NaN or infinite values")
    return np.asarray(tensor, dtype=np.float32)


def extract_mmt_features(
    waveform: np.ndarray, sample_rate: int, config: AudioConfig
) -> np.ndarray:
    """Build the paper-informed `(84, 2T, 1)` MFCC/Mel/Tonnetz tensor.

    The paper reports MFCC `(42,T)` doubled along time, Tonnetz `(6,T)` expanded
    to `(42,T)`, Mel `(42,T)`, and a final MMT shape `(84,2T)`, but it does not
    state the axis used for the first fusion. This geometry-consistent
    interpretation concatenates Tonnetz then Mel along time to `(42,2T)`, and
    then concatenates that block with the time-doubled MFCC along features.
    This implementation detail is an explicit project interpretation, not a
    claim about an unreported paper detail.
    """

    config.validate()
    if config.feature_set != "mmt":
        raise AudioContractError("extract_mmt_features requires feature_set='mmt'")
    if sample_rate != config.sample_rate:
        raise AudioContractError(
            f"feature extraction expected sample rate {config.sample_rate}, got {sample_rate}"
        )
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0:
        raise AudioContractError("feature extraction expects a non-empty mono waveform")
    if not np.isfinite(audio).all():
        raise AudioContractError("waveform contains NaN or infinite samples")
    if float(np.max(np.abs(audio))) <= np.finfo(np.float32).eps:
        raise AudioContractError("cannot extract features from silent audio")

    feature_audio = audio
    if feature_audio.size < config.n_fft:
        feature_audio = np.pad(
            feature_audio, (0, config.n_fft - feature_audio.size), mode="constant"
        )

    try:
        mel_power = librosa.feature.melspectrogram(
            y=feature_audio,
            sr=sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=42,
            power=2.0,
        ).astype(np.float32, copy=False)
        log_mel = librosa.power_to_db(mel_power, ref=np.max).astype(
            np.float32, copy=False
        )
        mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=42).astype(
            np.float32, copy=False
        )
        tonnetz = librosa.feature.tonnetz(y=feature_audio, sr=sample_rate).astype(
            np.float32, copy=False
        )
    except Exception as exc:
        raise AudioContractError(f"could not extract MMT features: {exc}") from exc

    if mfcc.ndim != 2 or mfcc.shape[0] != 42:
        raise AudioContractError(f"MFCC must have shape (42, frames), got {mfcc.shape}")
    if mel_power.ndim != 2 or mel_power.shape[0] != 42:
        raise AudioContractError(
            f"Mel spectrogram must have shape (42, frames), got {mel_power.shape}"
        )
    if tonnetz.ndim != 2 or tonnetz.shape[0] != 6:
        raise AudioContractError(f"Tonnetz must have shape (6, frames), got {tonnetz.shape}")

    frames = config.mmt_frames
    mfcc = _fit_time_axis(mfcc, frames)
    mel_power = _fit_time_axis(mel_power, frames)
    tonnetz = _fit_time_axis(tonnetz, frames)
    tonnetz_expanded = np.repeat(tonnetz, repeats=7, axis=0)
    mfcc_time_doubled = np.tile(mfcc, reps=(1, 2))
    tonnetz_then_mel = np.concatenate((tonnetz_expanded, mel_power), axis=1)
    features = np.concatenate((mfcc_time_doubled, tonnetz_then_mel), axis=0)
    if features.shape != config.tensor_shape[:2]:
        raise AudioContractError(
            f"internal MMT shape mismatch: expected {config.tensor_shape[:2]}, got {features.shape}"
        )
    tensor = features[:, :, np.newaxis]
    if not np.isfinite(tensor).all():
        raise AudioContractError("MMT extraction produced NaN or infinite values")
    return np.asarray(tensor, dtype=np.float32)


def _largest_odd_at_most(value: int) -> int:
    return value if value % 2 else value - 1


def _ensure_delta_frames(features: np.ndarray) -> np.ndarray:
    if features.shape[1] >= 3:
        return features
    return np.pad(features, ((0, 0), (0, 3 - features.shape[1])), mode="edge")


def _fit_minimum_frames(features: np.ndarray, frame_count: int) -> np.ndarray:
    if features.shape[1] >= frame_count:
        return features[:, :frame_count]
    return np.pad(
        features,
        ((0, 0), (0, frame_count - features.shape[1])),
        mode="edge",
    )


def _fit_time_axis(features: np.ndarray, max_frames: int) -> np.ndarray:
    frames = features.shape[1]
    if frames >= max_frames:
        return features[:, :max_frames]
    return np.pad(features, ((0, 0), (0, max_frames - frames)), mode="constant")
