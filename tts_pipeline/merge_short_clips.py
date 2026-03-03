#!/usr/bin/env python3
"""
merge_short_clips.py — Concatenate short clips from the same source file
=========================================================================
Groups Stage 1 output clips by their original source file, then
concatenates clips shorter than --min-duration into longer segments,
keeping clips from different source files separate.

Clips already >= min_duration are saved as-is.

CLI usage:
    python merge_short_clips.py --input ./output/cleaned --output ./output/merged
    python merge_short_clips.py --input ./output/cleaned --output ./output/merged --min-duration 10 --max-duration 15
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_utils import setup_logger

# ── Helpers ────────────────────────────────────────────────────────────────

SR = 22050  # must match stage1 output


def load_wav(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio


def silence_pad(ms: int = 300) -> np.ndarray:
    return np.zeros(int(SR * ms / 1000), dtype=np.float32)


def save_wav(audio: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(path), audio, SR, subtype="PCM_16")


def get_group_key(stem: str) -> str:
    """
    Extract the source-file key from a Stage 1 output filename.

    Stage 1 names clips as:  {original_stem}_{NNNN}
    e.g.  cerebral_rec01_0003  →  group key = cerebral_rec01
          expr_audio_0000      →  group key = expr_audio

    If the last part is not a 4-digit index, treat the whole stem as the key.
    """
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
        return parts[0]
    return stem


def flush_buffer(
    buffer: list[np.ndarray],
    group_key: str,
    chunk_idx: int,
    output_dir: Path,
    silence_ms: int,
) -> int:
    """Concatenate buffer and save. Returns 1 on success, 0 on empty buffer."""
    if not buffer:
        return 0
    pieces = []
    pad = silence_pad(silence_ms)
    for i, chunk in enumerate(buffer):
        if i > 0:
            pieces.append(pad)
        pieces.append(chunk)
    combined = np.concatenate(pieces)
    out_path = output_dir / f"{group_key}_{chunk_idx:04d}.wav"
    save_wav(combined, out_path)
    return 1


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge short clips from the same source into longer segments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",        "-i", required=True,
                   help="Folder of Stage 1 cleaned clips.")
    p.add_argument("--output",       "-o", required=True,
                   help="Output folder for merged clips.")
    p.add_argument("--min-duration", type=float, default=10.0,
                   help="Clips shorter than this (seconds) will be concatenated.")
    p.add_argument("--max-duration", type=float, default=15.0,
                   help="Maximum duration of merged output clips (seconds).")
    p.add_argument("--silence-ms",   type=int,   default=300,
                   help="Silence padding between merged clips (ms).")
    p.add_argument("--log-dir",      default="./logs")
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    logger     = setup_logger("merge_short_clips", log_dir=args.log_dir)
    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    min_dur    = args.min_duration
    max_dur    = args.max_duration
    sil_ms     = args.silence_ms

    output_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(input_dir.glob("*.wav"))
    if not wavs:
        logger.error(f"No WAV files found in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(wavs)} clips. min={min_dur}s  max={max_dur}s")

    # ── Group by source file ───────────────────────────────────────────────
    groups: dict[str, list[Path]] = defaultdict(list)
    for wav in wavs:
        key = get_group_key(wav.stem)
        groups[key].append(wav)

    logger.info(f"Grouped into {len(groups)} source files.")

    # ── Process each group ─────────────────────────────────────────────────
    total_saved   = 0
    total_input   = 0
    short_merged  = 0
    kept_as_is    = 0

    for group_key, files in tqdm(groups.items(), desc="Merging", unit="group"):
        files = sorted(files)
        buffer: list[np.ndarray] = []
        buffer_dur = 0.0
        chunk_idx  = 0

        for wav_path in files:
            try:
                audio = load_wav(wav_path)
            except Exception as exc:
                logger.warning(f"Cannot load {wav_path.name}: {exc} — skipping.")
                continue

            dur = len(audio) / SR
            total_input += 1

            if dur >= min_dur:
                # Long enough — flush any pending buffer first, then save as-is
                if buffer:
                    total_saved += flush_buffer(
                        buffer, group_key, chunk_idx, output_dir, sil_ms
                    )
                    chunk_idx += 1
                    buffer = []
                    buffer_dur = 0.0
                out_path = output_dir / f"{group_key}_{chunk_idx:04d}.wav"
                save_wav(audio, out_path)
                chunk_idx  += 1
                total_saved += 1
                kept_as_is  += 1

            else:
                # Short clip — try to merge with neighbours
                sil_dur = sil_ms / 1000
                projected = buffer_dur + (sil_dur if buffer else 0) + dur

                if buffer and projected > max_dur:
                    # Flush buffer before adding this clip
                    total_saved += flush_buffer(
                        buffer, group_key, chunk_idx, output_dir, sil_ms
                    )
                    chunk_idx  += 1
                    short_merged += 1
                    buffer     = [audio]
                    buffer_dur = dur
                else:
                    buffer.append(audio)
                    buffer_dur += dur + (sil_dur if len(buffer) > 1 else 0)

        # Final flush for this group
        if buffer:
            total_saved += flush_buffer(
                buffer, group_key, chunk_idx, output_dir, sil_ms
            )
            short_merged += 1

    logger.info("=" * 60)
    logger.info(f"Input clips   : {total_input}")
    logger.info(f"Output clips  : {total_saved}  "
                f"(kept as-is: {kept_as_is} | merged groups: {short_merged})")
    logger.info(f"Reduction     : {total_input} → {total_saved} clips")
    logger.info(f"Output        : {output_dir.resolve()}")


if __name__ == "__main__":
    main()
