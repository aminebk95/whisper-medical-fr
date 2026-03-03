#!/usr/bin/env python3
"""
run_pipeline.py — End-to-end TTS Pipeline Orchestrator
=======================================================
Runs all stages (or a subset) in sequence using settings from config.yaml.

CLI usage:
    python run_pipeline.py --config config.yaml
    python run_pipeline.py --config config.yaml --stages 1 4 5
    python run_pipeline.py --config config.yaml --stages 1 2 3 4 5 --dry-run

Stage map:
    1 → stage1_clean.py
    2 → stage2_concat.py
    3 → stage3_generate.py  (requires --texts for Stage 3)
    4 → stage4_prepare.py
    5 → stage5_train.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml

# ── Logging (simple, no utils dependency here) ─────────────────────────────
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | pipeline | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pipeline")


# ── Helpers ────────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_stage(
    cmd: list[str],
    stage_name: str,
    dry_run: bool = False,
) -> bool:
    """
    Execute *cmd* as a subprocess.

    Returns True on success, False on failure (non-zero exit code).
    """
    log.info(f"{'[DRY-RUN] ' if dry_run else ''}Starting {stage_name}")
    log.info(f"  Command: {' '.join(str(c) for c in cmd)}")

    if dry_run:
        return True

    start = time.time()
    result = subprocess.run(cmd, text=True)
    elapsed = timedelta(seconds=int(time.time() - start))

    if result.returncode == 0:
        log.info(f"  {stage_name} finished in {elapsed}  ✓")
        return True
    else:
        log.error(f"  {stage_name} FAILED (exit code {result.returncode})  ✗")
        return False


def py(script: str) -> list[str]:
    """Return [sys.executable, script] — ensures same Python env is used."""
    return [sys.executable, str(Path(__file__).parent / script)]


# ── Stage builders ─────────────────────────────────────────────────────────

def build_stage1_cmd(cfg: dict, config_path: Path) -> list[str]:
    paths = cfg["paths"]
    return py("stage1_clean.py") + [
        "--input",   paths["raw_input"],
        "--output",  paths["cleaned"],
        "--config",  str(config_path),
        "--log-dir", paths["logs"],
    ]


def build_stage2_cmd(cfg: dict, config_path: Path, manifest: str | None) -> list[str]:
    paths = cfg["paths"]
    cmd = py("stage2_concat.py") + [
        "--input",   paths["cleaned"],
        "--output",  paths["concatenated"],
        "--config",  str(config_path),
        "--log-dir", paths["logs"],
    ]
    if manifest:
        cmd += ["--manifest", manifest]
    return cmd


def build_stage3_cmd(cfg: dict, config_path: Path, texts_file: str) -> list[str]:
    paths = cfg["paths"]
    return py("stage3_generate.py") + [
        "--input",   texts_file,
        "--output",  paths["generated"],
        "--config",  str(config_path),
        "--log-dir", paths["logs"],
    ]


def build_stage4_cmd(
    cfg: dict, config_path: Path, stage3_ran: bool, no_transcribe: bool
) -> list[str]:
    paths = cfg["paths"]
    # If Stage 3 ran, input is the generated folder; otherwise, use cleaned
    input_dir = paths["generated"] if stage3_ran else paths["cleaned"]
    cmd = py("stage4_prepare.py") + [
        "--input",   input_dir,
        "--output",  paths["dataset"],
        "--config",  str(config_path),
        "--log-dir", paths["logs"],
    ]
    if no_transcribe:
        cmd.append("--no-transcribe")
    return cmd


def build_stage5_cmd(
    cfg: dict, config_path: Path, resume: str | None
) -> list[str]:
    paths  = cfg["paths"]
    cmd = py("stage5_train.py") + [
        "--dataset", paths["dataset"],
        "--output",  paths["training"],
        "--config",  str(config_path),
        "--log-dir", paths["logs"],
    ]
    if resume:
        cmd += ["--resume", resume]
    return cmd


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TTS Pipeline Orchestrator — runs all stages end-to-end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", "-c", default="config.yaml",
                   help="Master config.yaml.")
    p.add_argument("--stages", nargs="+", type=int,
                   default=[1, 4, 5],
                   help="Stages to run (e.g. --stages 1 4 5). "
                        "Stage 3 requires --texts. Stage 2 is optional.")
    p.add_argument("--texts", default=None,
                   help="Path to texts.txt/csv for Stage 3 (required if running stage 3).")
    p.add_argument("--manifest", default=None,
                   help="Optional clip order file for Stage 2.")
    p.add_argument("--resume", default=None,
                   help="Checkpoint path for Stage 5 resume.")
    p.add_argument("--no-transcribe", action="store_true",
                   help="Pass --no-transcribe to Stage 4 (skip Whisper).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing them.")
    return p.parse_args()


def main() -> None:
    args        = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        log.error(f"Config file not found: {config_path}")
        sys.exit(1)

    cfg    = load_config(config_path)
    stages = sorted(set(args.stages))

    log.info("=" * 60)
    log.info("TTS Pipeline — starting")
    log.info(f"Config : {config_path.resolve()}")
    log.info(f"Stages : {stages}")
    log.info("=" * 60)

    # Validate prerequisites
    if 3 in stages and not args.texts:
        log.error("Stage 3 requires --texts <file>. Aborting.")
        sys.exit(1)

    # Create output dirs
    paths = cfg.get("paths", {})
    for key, val in paths.items():
        if key != "logs":
            Path(val).mkdir(parents=True, exist_ok=True)
    Path(paths.get("logs", "./logs")).mkdir(parents=True, exist_ok=True)

    # Track which stages ran (for stage4 input path decision)
    stage3_ran = False
    failed_stages: list[int] = []
    pipeline_start = time.time()

    for stage_num in stages:
        if stage_num == 1:
            cmd = build_stage1_cmd(cfg, config_path)
            ok  = run_stage(cmd, "Stage 1 — Audio Cleaning", dry_run=args.dry_run)

        elif stage_num == 2:
            cmd = build_stage2_cmd(cfg, config_path, args.manifest)
            ok  = run_stage(cmd, "Stage 2 — Concatenation", dry_run=args.dry_run)

        elif stage_num == 3:
            cmd = build_stage3_cmd(cfg, config_path, args.texts)
            ok  = run_stage(cmd, "Stage 3 — TTS Generation", dry_run=args.dry_run)
            if ok:
                stage3_ran = True

        elif stage_num == 4:
            cmd = build_stage4_cmd(cfg, config_path, stage3_ran, args.no_transcribe)
            ok  = run_stage(cmd, "Stage 4 — Dataset Preparation", dry_run=args.dry_run)

        elif stage_num == 5:
            cmd = build_stage5_cmd(cfg, config_path, args.resume)
            ok  = run_stage(cmd, "Stage 5 — Model Training", dry_run=args.dry_run)

        else:
            log.warning(f"Unknown stage number {stage_num} — skipping.")
            ok = True

        if not ok:
            failed_stages.append(stage_num)
            log.error(f"Pipeline halted at Stage {stage_num}. Fix errors above.")
            break

    elapsed = timedelta(seconds=int(time.time() - pipeline_start))
    log.info("=" * 60)
    if not failed_stages:
        log.info(f"Pipeline completed successfully in {elapsed}.")
    else:
        log.error(f"Pipeline failed at stage(s): {failed_stages}. Total elapsed: {elapsed}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
