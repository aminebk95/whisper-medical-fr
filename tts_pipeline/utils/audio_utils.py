"""Shared audio helper functions for the TTS pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import librosa
import soundfile as sf


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_audio(path: str | Path, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Load any audio file, convert to mono, resample to *target_sr*.

    Returns:
        (audio_array_float32, sample_rate)
    """
    audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return audio.astype(np.float32), sr


def save_audio(audio: np.ndarray, path: str | Path, sr: int = 22050) -> None:
    """Save a float32 numpy array as a 16-bit PCM WAV file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Clip to [-1, 1] before writing to avoid clipping artefacts
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(out), audio, sr, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_rms(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Scale audio so its RMS equals *target_db* dBFS."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-8:
        return audio
    target_rms = 10 ** (target_db / 20.0)
    return audio * (target_rms / rms)


def normalize_peak(audio: np.ndarray, headroom_db: float = -1.0) -> np.ndarray:
    """Scale audio so its peak equals *headroom_db* dBFS."""
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    target_peak = 10 ** (headroom_db / 20.0)
    return audio * (target_peak / peak)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_duration(audio: np.ndarray, sr: int) -> float:
    """Duration in seconds."""
    return len(audio) / sr


def compute_rms_db(audio: np.ndarray) -> float:
    """RMS level in dBFS."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return -100.0
    return float(20.0 * np.log10(rms))


def compute_snr(audio: np.ndarray, sr: int, noise_percentile: float = 10.0) -> float:
    """
    Rough SNR estimate: signal RMS vs. the quietest *noise_percentile*% of frames.

    Returns SNR in dB (capped at 60 dB if the noise floor is below the threshold).
    """
    frame_len = int(sr * 0.025)   # 25 ms frames
    hop_len   = int(sr * 0.010)   # 10 ms hop
    frames    = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop_len)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=0))
    noise_floor = np.percentile(frame_rms, noise_percentile)
    signal_rms  = np.sqrt(np.mean(audio ** 2))
    if noise_floor < 1e-10:
        return 60.0
    return float(20.0 * np.log10(max(signal_rms, 1e-10) / noise_floor))
