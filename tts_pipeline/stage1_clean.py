#!/usr/bin/env python3
"""
Stage 1 — Audio Cleaning
========================
Converts raw audio (wav/mp3/flac/m4a/ogg) into clean, normalised,
denoised, VAD-segmented WAV clips ready for dataset preparation.

CLI usage:
    python stage1_clean.py --input ./raw --output ./output/cleaned
    python stage1_clean.py --input ./raw --output ./output/cleaned --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

# Lazy imports (some are heavy — only load when needed)

# ── Project utilities ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_utils import setup_logger
from utils.audio_utils import (
    load_audio,
    save_audio,
    normalize_rms,
    normalize_peak,
    get_duration,
    compute_rms_db,
    compute_snr,
)

logger: logging.Logger = None  # set in main()

# ── Defaults ───────────────────────────────────────────────────────────────
DEFAULT_CFG = {
    "sample_rate": 22050,
    "normalization": "rms",
    "target_rms_db": -20.0,
    "peak_headroom_db": -1.0,
    "noise_reduction": {
        "enabled": True,
        "prop_decrease": 0.75,
        "stationary": False,
    },
    "vad": {
        "method": "silero",
        "min_speech_duration_ms": 200,
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 100,
        "top_db": 35,
    },
    "segment_filter": {
        "min_duration_s": 1.0,
        "max_duration_s": 15.0,
    },
    "supported_formats": ["wav", "mp3", "flac", "m4a", "ogg"],
}


# ── Format conversion ──────────────────────────────────────────────────────

def convert_to_wav(src: Path, tmp_dir: Path) -> Path | None:
    """
    Convert *src* to a temporary WAV file using pydub / ffmpeg.
    Returns the WAV path, or None on failure.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        logger.error("pydub not installed — cannot convert non-WAV files.")
        return None

    tmp_dir.mkdir(parents=True, exist_ok=True)
    wav_path = tmp_dir / f"{src.stem}.wav"
    try:
        audio = AudioSegment.from_file(str(src))
        audio.export(str(wav_path), format="wav")
        return wav_path
    except Exception as exc:
        logger.warning(f"Could not convert {src.name}: {exc}")
        return None


# ── Noise reduction ────────────────────────────────────────────────────────

def apply_noise_reduction(audio: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """Apply noisereduce to *audio*. Returns original if library not found."""
    try:
        import noisereduce as nr
        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            prop_decrease=cfg.get("prop_decrease", 0.75),
            stationary=cfg.get("stationary", False),
        )
        return reduced.astype(np.float32)
    except ImportError:
        logger.warning("noisereduce not installed — skipping noise reduction.")
        return audio
    except Exception as exc:
        logger.warning(f"Noise reduction failed: {exc} — using original audio.")
        return audio


# ── VAD / segment detection ────────────────────────────────────────────────

def _get_segments_silero(
    audio: np.ndarray, sr: int, vad_cfg: dict
) -> List[Tuple[int, int]]:
    """
    Use Silero VAD to get list of (start_sample, end_sample) speech intervals.
    Falls back to librosa on failure.
    """
    try:
        import torch
        from silero_vad import load_silero_vad, get_speech_timestamps

        model = load_silero_vad()
        # Silero expects 16 kHz; resample if needed
        if sr != 16000:
            import librosa as _librosa
            audio_16k = _librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio

        tensor = torch.FloatTensor(audio_16k)
        timestamps = get_speech_timestamps(
            tensor,
            model,
            sampling_rate=16000,
            min_speech_duration_ms=vad_cfg.get("min_speech_duration_ms", 200),
            min_silence_duration_ms=vad_cfg.get("min_silence_duration_ms", 500),
            speech_pad_ms=vad_cfg.get("speech_pad_ms", 100),
            return_seconds=False,
        )

        scale = sr / 16000  # convert 16k samples → original sr
        return [(int(t["start"] * scale), int(t["end"] * scale)) for t in timestamps]

    except Exception as exc:
        logger.warning(f"Silero VAD failed ({exc}) — falling back to librosa.")
        return _get_segments_librosa(audio, sr, vad_cfg)


def _get_segments_librosa(
    audio: np.ndarray, sr: int, vad_cfg: dict
) -> List[Tuple[int, int]]:
    """Use librosa.effects.split for silence-based segmentation."""
    import librosa
    intervals = librosa.effects.split(audio, top_db=vad_cfg.get("top_db", 35))
    return [(int(s), int(e)) for s, e in intervals]


def get_speech_segments(
    audio: np.ndarray, sr: int, vad_cfg: dict
) -> List[Tuple[int, int]]:
    method = vad_cfg.get("method", "silero")
    if method == "silero":
        return _get_segments_silero(audio, sr, vad_cfg)
    return _get_segments_librosa(audio, sr, vad_cfg)


# ── Segment helpers ────────────────────────────────────────────────────────

def merge_close_segments(
    segments: List[Tuple[int, int]],
    sr: int,
    max_gap_ms: int = 500,
) -> List[Tuple[int, int]]:
    """Merge consecutive segments separated by less than *max_gap_ms* ms."""
    if not segments:
        return segments
    max_gap_samples = int(sr * max_gap_ms / 1000)
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap_samples:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def filter_segments(
    segments: List[Tuple[int, int]],
    sr: int,
    min_s: float,
    max_s: float,
) -> List[Tuple[int, int]]:
    """Remove segments outside the [min_s, max_s] duration range."""
    min_samples = int(sr * min_s)
    max_samples = int(sr * max_s)
    return [(s, e) for s, e in segments if min_samples <= (e - s) <= max_samples]


def subsplit_long_segment(
    audio: np.ndarray,
    start: int,
    end: int,
    sr: int,
    max_s: float,
    top_db: float = 35,
) -> List[Tuple[int, int]]:
    """
    Recursively split a segment that is longer than *max_s* seconds
    by finding the loudest silence point inside it.
    """
    import librosa

    duration = (end - start) / sr
    if duration <= max_s:
        return [(start, end)]

    chunk = audio[start:end]
    sub_intervals = librosa.effects.split(chunk, top_db=top_db)

    if len(sub_intervals) <= 1:
        # No internal silence — cut at mid-point
        mid = (start + end) // 2
        return (
            subsplit_long_segment(audio, start, mid, sr, max_s, top_db)
            + subsplit_long_segment(audio, mid, end, sr, max_s, top_db)
        )

    # Recursively handle each sub-interval, offset by 'start'
    result: List[Tuple[int, int]] = []
    for s, e in sub_intervals:
        result.extend(
            subsplit_long_segment(audio, start + s, start + e, sr, max_s, top_db)
        )
    return result


# ── Per-file processing ────────────────────────────────────────────────────

def process_file(
    src: Path,
    output_dir: Path,
    cfg: dict,
    tmp_dir: Path,
) -> Tuple[int, int]:
    """
    Process a single source audio file.

    Returns:
        (segments_saved, segments_skipped)
    """
    sr = cfg["sample_rate"]
    min_s = cfg["segment_filter"]["min_duration_s"]
    max_s = cfg["segment_filter"]["max_duration_s"]

    # 1. Convert to WAV if necessary
    if src.suffix.lower() != ".wav":
        wav_src = convert_to_wav(src, tmp_dir)
        if wav_src is None:
            logger.warning(f"Skipping {src.name} — conversion failed.")
            return 0, 1
    else:
        wav_src = src

    # 2. Load + resample + mono
    try:
        audio, _ = load_audio(wav_src, target_sr=sr)
    except Exception as exc:
        logger.warning(f"Skipping {src.name} — load failed: {exc}")
        return 0, 1

    if len(audio) == 0:
        logger.warning(f"Skipping {src.name} — empty audio.")
        return 0, 1

    # 3. Normalise
    if cfg["normalization"] == "rms":
        audio = normalize_rms(audio, target_db=cfg["target_rms_db"])
    else:
        audio = normalize_peak(audio, headroom_db=cfg["peak_headroom_db"])

    # 4. Noise reduction
    if cfg["noise_reduction"]["enabled"]:
        audio = apply_noise_reduction(audio, sr, cfg["noise_reduction"])

    # 5. Re-normalise after denoising (denoiser can change levels)
    if cfg["normalization"] == "rms":
        audio = normalize_rms(audio, target_db=cfg["target_rms_db"])
    else:
        audio = normalize_peak(audio, headroom_db=cfg["peak_headroom_db"])

    # 6. VAD — detect speech segments
    segments = get_speech_segments(audio, sr, cfg["vad"])
    segments = merge_close_segments(segments, sr, max_gap_ms=500)

    # 7. Sub-split segments that are still too long
    expanded: List[Tuple[int, int]] = []
    for s, e in segments:
        if (e - s) / sr > max_s:
            expanded.extend(
                subsplit_long_segment(audio, s, e, sr, max_s,
                                      top_db=cfg["vad"].get("top_db", 35))
            )
        else:
            expanded.append((s, e))

    # 8. Filter by duration
    filtered = filter_segments(expanded, sr, min_s, max_s)

    if not filtered:
        logger.warning(f"No valid segments found in {src.name}.")
        return 0, 1

    # 9. Save segments
    stem = src.stem
    saved = 0
    for idx, (s, e) in enumerate(filtered):
        segment = audio[s:e]
        out_name = f"{stem}_{idx:04d}.wav"
        out_path = output_dir / out_name
        try:
            save_audio(segment, out_path, sr=sr)
            dur = get_duration(segment, sr)
            rms = compute_rms_db(segment)
            snr = compute_snr(segment, sr)
            logger.debug(
                f"  Saved {out_name}  dur={dur:.2f}s  RMS={rms:.1f}dBFS  SNR={snr:.1f}dB"
            )
            saved += 1
        except Exception as exc:
            logger.warning(f"  Could not save {out_name}: {exc}")

    return saved, len(expanded) - len(filtered)


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1: Audio Cleaning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  "-i", required=True, help="Folder with raw audio files.")
    p.add_argument("--output", "-o", required=True, help="Folder for cleaned segments.")
    p.add_argument("--config", "-c", default=None,  help="Path to config.yaml.")
    p.add_argument("--log-dir", default="./logs",   help="Directory for log files.")
    p.add_argument("--sample-rate", type=int, default=None, help="Override sample rate.")
    p.add_argument("--no-denoise", action="store_true", help="Skip noise reduction.")
    return p.parse_args()


def load_config(path: str | None) -> dict:
    cfg = DEFAULT_CFG.copy()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            full = yaml.safe_load(f)
        if "stage1" in full:
            # Merge only the stage1 section (deep merge for nested dicts)
            def _merge(base: dict, override: dict) -> dict:
                out = base.copy()
                for k, v in override.items():
                    if isinstance(v, dict) and isinstance(out.get(k), dict):
                        out[k] = _merge(out[k], v)
                    else:
                        out[k] = v
                return out
            cfg = _merge(cfg, full["stage1"])
    return cfg


def main() -> None:
    global logger
    args = parse_args()
    logger = setup_logger("stage1_clean", log_dir=args.log_dir)

    cfg = load_config(args.config)
    if args.sample_rate:
        cfg["sample_rate"] = args.sample_rate
    if args.no_denoise:
        cfg["noise_reduction"]["enabled"] = False

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    tmp_dir    = output_dir / "_tmp"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Collect all supported audio files
    supported = set(cfg.get("supported_formats", ["wav", "mp3", "flac", "m4a", "ogg"]))
    files = [
        f for f in sorted(input_dir.rglob("*"))
        if f.is_file() and f.suffix.lstrip(".").lower() in supported
    ]

    if not files:
        logger.error(f"No audio files found in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(files)} audio file(s) in {input_dir}")
    logger.info(f"Output → {output_dir}")

    try:
        from tqdm import tqdm
        file_iter = tqdm(files, desc="Cleaning", unit="file")
    except ImportError:
        file_iter = files

    total_saved = 0
    total_skipped = 0

    for src in file_iter:
        logger.info(f"Processing: {src.name}")
        saved, skipped = process_file(src, output_dir, cfg, tmp_dir)
        total_saved   += saved
        total_skipped += skipped
        logger.info(f"  → {saved} segment(s) saved, {skipped} skipped.")

    # Clean up temp dir
    if tmp_dir.exists():
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("=" * 60)
    logger.info(f"Stage 1 complete. Segments saved: {total_saved} | Skipped: {total_skipped}")
    logger.info(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
