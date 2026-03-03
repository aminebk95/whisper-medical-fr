#!/usr/bin/env python3
"""
Stage 5 — Model Training
=========================
Trains (or fine-tunes) a TTS model on the LJSpeech dataset produced
by Stage 4.

Supported models:
  vits  → Trains from scratch or fine-tunes VITS.
          Works on GTX 1650 (4 GB VRAM) with fp16 + batch_size 8.  ✅
  xtts  → Fine-tunes XTTS v2.
          Requires ≥ 6 GB VRAM. Will OOM on GTX 1650.              ⚠️

CLI usage:
    python stage5_train.py --dataset ./output/dataset --output ./output/training
    python stage5_train.py --dataset ./output/dataset --output ./output/training --model vits
    python stage5_train.py --dataset ./output/dataset --output ./output/training --resume ./output/training/best_model.pth
    python stage5_train.py --dataset ./output/dataset --output ./output/training --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_utils import setup_logger

logger: logging.Logger = None

# ── Defaults ───────────────────────────────────────────────────────────────
DEFAULT_CFG = {
    "sample_rate": 22050,
    "model": "vits",
    "tensorboard": True,
    "save_best_model": True,
    "random_seed": 42,
    "vits": {
        "base_model": None,
        "phoneme_language": "fr-fr",
        "use_phonemes": True,
        "batch_size": 8,
        "eval_batch_size": 8,
        "grad_clip": 1000.0,
        "learning_rate": 2.0e-4,
        "betas": [0.8, 0.99],
        "eps": 1.0e-9,
        "lr_decay": 0.999875,
        "epochs": 1000,
        "checkpoint_every_n_steps": 1000,
        "eval_every_n_steps": 500,
        "mixed_precision": True,
        "num_loader_workers": 2,
        "test_sentences": [
            "Le patient présente une tachycardie sinusale à quatre-vingt-dix battements par minute.",
            "La pression artérielle est de cent vingt sur quatre-vingts millimètres de mercure.",
        ],
    },
    "xtts": {
        "base_model": "tts_models/multilingual/multi-dataset/xtts_v2",
        "fine_tune": True,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 5.0e-6,
        "epochs": 50,
        "checkpoint_every_n_steps": 500,
        "fp16": True,
        "gradient_checkpointing": True,
        "num_loader_workers": 0,
        "speaker_name": "voice_fr_medical",
        "test_sentence": "Le patient présente une tachycardie sinusale.",
    },
}


# ── Hardware check ─────────────────────────────────────────────────────────

def check_hardware(model: str) -> dict:
    """Return dict with GPU info and warnings."""
    info = {"device": "cpu", "vram_gb": 0.0, "warnings": []}
    try:
        import torch
        if torch.cuda.is_available():
            info["device"] = "cuda"
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = props.total_memory / (1024 ** 3)
            info["gpu_name"] = props.name
            if model == "xtts" and info["vram_gb"] < 6.0:
                info["warnings"].append(
                    f"XTTS v2 requires ≥6 GB VRAM. You have {info['vram_gb']:.1f} GB "
                    f"({props.name}). Training will likely OOM. "
                    "Consider using --model vits instead, or rent a cloud GPU."
                )
        else:
            info["warnings"].append("CUDA not available — training will be very slow on CPU.")
    except ImportError:
        info["warnings"].append("PyTorch not found.")
    return info


# ── VITS training ──────────────────────────────────────────────────────────

def train_vits(dataset_dir: Path, output_dir: Path, cfg: dict, resume: str | None) -> None:
    """
    Train VITS using the Coqui TTS trainer.
    """
    try:
        from TTS.tts.configs.vits_config import VitsConfig
        from TTS.tts.models.vits import Vits
        from TTS.tts.datasets import load_tts_samples
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
        from TTS.utils.audio.processor import AudioProcessor
        from trainer import Trainer, TrainerArgs
    except ImportError as exc:
        logger.error(
            f"Coqui TTS not installed: {exc}\n"
            "Install it with: pip install TTS"
        )
        sys.exit(1)

    vcfg = cfg["vits"]
    sr   = cfg["sample_rate"]

    from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata.csv",
        path=str(dataset_dir),
        language=cfg.get("language", "fr"),
    )

    audio_config = BaseAudioConfig(
        sample_rate=sr,
        do_trim_silence=True,
        trim_db=45,
        resample=False,
        mel_fmin=0.0,
        mel_fmax=None,
    )

    vits_config = VitsConfig(
        audio=audio_config,
        run_name="vits_fr_medical",
        batch_size=vcfg["batch_size"],
        eval_batch_size=vcfg["eval_batch_size"],
        batch_group_size=5,
        num_loader_workers=vcfg["num_loader_workers"],
        num_eval_loader_workers=max(1, vcfg["num_loader_workers"] // 2),
        run_eval=True,
        test_delay_epochs=-1,
        epochs=vcfg["epochs"],
        save_step=vcfg["checkpoint_every_n_steps"],
        save_n_checkpoints=5,
        save_best_after=vcfg["checkpoint_every_n_steps"],
        save_checkpoints=True,
        print_step=25,
        print_eval=True,
        mixed_precision=vcfg["mixed_precision"],
        output_path=str(output_dir),
        datasets=[dataset_config],
        cudnn_benchmark=False,
        # Phoneme settings
        use_phonemes=vcfg["use_phonemes"],
        phoneme_language=vcfg["phoneme_language"],
        phoneme_cache_path=str(output_dir / "phoneme_cache"),
        compute_input_seq_cache=True,
        # Optimiser
        lr=vcfg["learning_rate"],
        betas=tuple(vcfg["betas"]),
        eps=vcfg["eps"],
        lr_decay=vcfg["lr_decay"],
        grad_clip=vcfg["grad_clip"],
        # Test sentences played after each eval
        test_sentences=vcfg.get("test_sentences", []),
    )

    # Optionally load a pretrained checkpoint
    checkpoint_path = resume or vcfg.get("base_model")

    # Build model & trainer
    ap       = AudioProcessor.init_from_config(vits_config)
    tokenizer, vits_config = TTSTokenizer.init_from_config(vits_config)
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=vcfg["eval_batch_size"] * 4,
        eval_split_size=0.05,
    )

    model = Vits(vits_config, ap, tokenizer, speaker_manager=None)

    trainer_args = TrainerArgs(
        restore_path=checkpoint_path,
        skip_train_epoch=False,
    )

    trainer = Trainer(
        args    = trainer_args,
        config  = vits_config,
        output_path = str(output_dir),
        model   = model,
        train_samples = train_samples,
        eval_samples  = eval_samples,
    )

    logger.info("Starting VITS training…")
    trainer.fit()
    logger.info("VITS training complete.")


# ── XTTS fine-tuning ───────────────────────────────────────────────────────

def train_xtts(dataset_dir: Path, output_dir: Path, cfg: dict, resume: str | None) -> None:
    """
    Fine-tune XTTS v2.

    NOTE: This requires ≥6 GB VRAM. On GTX 1650 (4 GB) it will OOM.
    If you must run on 4 GB, try:
        --model vits
    or rent a cloud GPU (e.g. Vast.ai, RunPod, Google Colab A100).
    """
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from TTS.config.shared_configs import BaseDatasetConfig
        from trainer import Trainer, TrainerArgs
    except ImportError as exc:
        logger.error(f"Coqui TTS not installed: {exc}\nInstall: pip install TTS")
        sys.exit(1)

    xcfg = cfg["xtts"]
    sr   = cfg["sample_rate"]

    # XTTS requires a reference speaker audio for the fine-tuning recipe
    # Collect a reference clip from the dataset
    wavs = sorted((dataset_dir / "wavs").glob("*.wav"))
    if not wavs:
        logger.error("No WAV files found in dataset/wavs/")
        sys.exit(1)
    speaker_ref = str(wavs[0])

    # --- Load base model config ---
    import torch
    from TTS.utils.manage import ModelManager

    logger.info(f"Downloading / loading base model: {xcfg['base_model']}")
    manager = ModelManager()
    model_path, config_path, _ = manager.download_model(xcfg["base_model"])

    base_config = XttsConfig()
    base_config.load_json(config_path)

    # Override training parameters for memory efficiency
    base_config.batch_size              = xcfg["batch_size"]
    base_config.eval_batch_size         = 1
    base_config.num_loader_workers      = xcfg["num_loader_workers"]
    base_config.grad_clip               = 1.0
    base_config.lr                      = xcfg["learning_rate"]
    base_config.epochs                  = xcfg["epochs"]
    base_config.save_step               = xcfg["checkpoint_every_n_steps"]
    base_config.mixed_precision         = xcfg.get("fp16", True)
    base_config.output_path             = str(output_dir)

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata.csv",
        path=str(dataset_dir),
        language=cfg.get("language", "fr"),
    )
    base_config.datasets = [dataset_config]

    model = Xtts.init_from_config(base_config)
    model.load_checkpoint(
        base_config,
        checkpoint_path=resume if resume else model_path,
        use_deepspeed=False,
    )

    if xcfg.get("gradient_checkpointing"):
        try:
            model.gpt.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")
        except AttributeError:
            logger.warning("Gradient checkpointing not available for this model version.")

    trainer_args = TrainerArgs(
        restore_path=resume,
        skip_train_epoch=False,
        grad_accum_steps=xcfg.get("gradient_accumulation_steps", 16),
    )

    from TTS.tts.datasets import load_tts_samples
    train_samples, eval_samples = load_tts_samples(
        dataset_config, eval_split=True, eval_split_size=0.05
    )

    trainer = Trainer(
        args        = trainer_args,
        config      = base_config,
        output_path = str(output_dir),
        model       = model,
        train_samples = train_samples,
        eval_samples  = eval_samples,
    )

    logger.info("Starting XTTS v2 fine-tuning…")
    trainer.fit()
    logger.info("XTTS v2 fine-tuning complete.")


# ── Sample inference ───────────────────────────────────────────────────────

def run_sample_inference(
    model_type: str,
    checkpoint_path: Path,
    test_sentence: str,
    output_dir: Path,
    cfg: dict,
) -> None:
    """Generate a sample WAV with the trained model to verify quality."""
    sample_path = output_dir / "sample_inference.wav"
    logger.info(f"Running sample inference → {sample_path}")

    try:
        from TTS.api import TTS
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_type == "vits":
            tts = TTS(model_path=str(checkpoint_path), config_path=str(checkpoint_path.parent / "config.json"))
            tts.tts_to_file(
                text=test_sentence,
                file_path=str(sample_path),
            )
        elif model_type == "xtts":
            # XTTS needs a reference speaker
            wavs = sorted((output_dir.parent / "dataset" / "wavs").glob("*.wav"))
            speaker_wav = str(wavs[0]) if wavs else None
            tts = TTS(model_path=str(checkpoint_path), config_path=str(checkpoint_path.parent / "config.json"))
            tts.tts_to_file(
                text=test_sentence,
                speaker_wav=speaker_wav,
                language=cfg.get("language", "fr"),
                file_path=str(sample_path),
            )

        logger.info(f"Sample saved: {sample_path}")
    except Exception as exc:
        logger.warning(f"Sample inference failed: {exc}")


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 5: TTS Model Training (VITS / XTTS v2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", "-d", required=True, help="LJSpeech dataset folder (Stage 4 output).")
    p.add_argument("--output",  "-o", required=True, help="Training output folder.")
    p.add_argument("--config",  "-c", default=None,  help="Path to config.yaml.")
    p.add_argument("--log-dir",       default="./logs")
    p.add_argument("--model", choices=["vits", "xtts"], default=None,
                   help="Model architecture (overrides config).")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume from.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch size.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override number of epochs.")
    p.add_argument("--check-hardware", action="store_true",
                   help="Print hardware info and VRAM warnings, then exit.")
    return p.parse_args()


def load_config(path: str | None) -> dict:
    cfg = DEFAULT_CFG.copy()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            full = yaml.safe_load(f)
        if "stage5" in full:
            def _merge(base: dict, override: dict) -> dict:
                out = base.copy()
                for k, v in override.items():
                    if isinstance(v, dict) and isinstance(out.get(k), dict):
                        out[k] = {**out[k], **v}
                    else:
                        out[k] = v
                return out
            cfg = _merge(cfg, full["stage5"])
        for key in ("sample_rate", "language"):
            if key in full:
                cfg.setdefault(key, full[key])
    return cfg


def main() -> None:
    global logger
    args   = parse_args()
    logger = setup_logger("stage5_train", log_dir=args.log_dir)

    cfg        = load_config(args.config)
    model_type = args.model or cfg.get("model", "vits")

    # Hardware check
    hw = check_hardware(model_type)
    for w in hw["warnings"]:
        logger.warning(f"[HW] {w}")
    if hw["device"] == "cuda":
        logger.info(f"GPU: {hw.get('gpu_name', 'unknown')}  VRAM: {hw['vram_gb']:.1f} GB")

    if args.check_hardware:
        return

    # Apply CLI overrides
    if args.batch_size is not None:
        cfg[model_type]["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg[model_type]["epochs"] = args.epochs

    dataset_dir = Path(args.dataset)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    # Validate dataset structure
    meta = dataset_dir / "metadata.csv"
    wavs = dataset_dir / "wavs"
    if not meta.exists():
        logger.error(f"metadata.csv not found in {dataset_dir}")
        sys.exit(1)
    if not wavs.exists():
        logger.error(f"wavs/ directory not found in {dataset_dir}")
        sys.exit(1)

    n_wavs = len(list(wavs.glob("*.wav")))
    logger.info(f"Dataset: {n_wavs} WAV files in {dataset_dir}")
    logger.info(f"Model  : {model_type.upper()}")
    logger.info(f"Output → {output_dir}")

    # XTTS VRAM warning
    if model_type == "xtts" and hw.get("vram_gb", 0) < 6.0:
        logger.error(
            "XTTS v2 requires ≥6 GB VRAM.  "
            f"Your GPU has {hw.get('vram_gb', 0):.1f} GB.  "
            "Run with --model vits or use a cloud GPU."
        )
        answer = input("Continue anyway? [y/N] ").strip().lower()
        if answer != "y":
            logger.info("Aborted by user.")
            sys.exit(0)

    # Dispatch to the right trainer
    if model_type == "vits":
        train_vits(dataset_dir, output_dir, cfg, resume=args.resume)
    elif model_type == "xtts":
        train_xtts(dataset_dir, output_dir, cfg, resume=args.resume)
    else:
        logger.error(f"Unknown model type: {model_type}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Stage 5 complete.")
    logger.info(f"Checkpoints → {output_dir.resolve()}")

    # Run a sample
    test_sentence = cfg.get(model_type, {}).get(
        "test_sentence",
        cfg.get(model_type, {}).get(
            "test_sentences",
            ["Le patient présente une tachycardie sinusale."]
        )
    )
    if isinstance(test_sentence, list):
        test_sentence = test_sentence[0]

    best_ckpt = output_dir / "best_model.pth"
    if best_ckpt.exists():
        run_sample_inference(model_type, best_ckpt, test_sentence, output_dir, cfg)


if __name__ == "__main__":
    main()
