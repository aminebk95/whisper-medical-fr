# TTS Dataset Pipeline & Training System
**Language**: French (Medical) | **Hardware**: GTX 1650 4 GB VRAM | **Model**: VITS (primary) / XTTS v2 (inference)

---

## Project Structure

```
tts_pipeline/
├── config.yaml            ← Master configuration (edit this)
├── requirements.txt       ← Python dependencies
├── run_pipeline.py        ← End-to-end orchestrator
│
├── stage1_clean.py        ← Audio cleaning & VAD segmentation
├── stage2_concat.py       ← Audio concatenation (optional)
├── stage3_generate.py     ← TTS generation with Edge-TTS (optional)
├── stage4_prepare.py      ← Dataset preparation (LJSpeech format)
├── stage5_train.py        ← VITS / XTTS v2 training
│
├── utils/
│   ├── audio_utils.py     ← Shared audio helpers
│   ├── text_utils.py      ← French medical text normalisation
│   └── logging_utils.py   ← Logging setup
│
├── raw/                   ← PUT YOUR RAW WAV FILES HERE
└── output/
    ├── cleaned/           ← Stage 1 output
    ├── concatenated/      ← Stage 2 output
    ├── generated/         ← Stage 3 output
    ├── dataset/           ← Stage 4 output (LJSpeech)
    └── training/          ← Stage 5 output (checkpoints)
```

---

## Quick Start

### 1. Install dependencies

```bash
# Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Install PyTorch with CUDA 11.8 (adjust for your CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install the rest
pip install -r requirements.txt
```

> **Also required (system-level):**
> - **FFmpeg**: `winget install ffmpeg` (Windows) or `sudo apt install ffmpeg`
> - **espeak-ng** (for VITS phonemizer): download from [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases)

### 2. Place your raw audio

Copy your WAV files into `tts_pipeline/raw/`.

### 3. Edit `config.yaml`

Key settings to review:
- `paths.raw_input` — point to your audio folder
- `stage1.noise_reduction.enabled` — set `false` if audio is already clean
- `stage4.whisper_model` — `medium` is a good balance; use `small` to save VRAM
- `stage5.model` — **keep `vits`** on a GTX 1650 (4 GB VRAM)

### 4. Run the pipeline

```bash
cd tts_pipeline

# Full pipeline (Stage 1 → 4 → 5)
python run_pipeline.py --config config.yaml --stages 1 4 5

# Or run stages individually (see below)
```

---

## Stage Reference

### Stage 1 — Audio Cleaning

Converts raw audio → clean 1–15s mono WAV clips at 22050 Hz.

**Steps:**
1. Format conversion (mp3/flac/m4a → wav)
2. Resample to 22050 Hz, convert to mono
3. RMS/peak normalisation
4. Noise reduction (noisereduce)
5. Silero VAD speech segmentation
6. Duration filtering (1–15 s)

```bash
python stage1_clean.py \
  --input  ./raw \
  --output ./output/cleaned \
  --config config.yaml

# Optional flags:
#   --sample-rate 22050   override sample rate
#   --no-denoise          skip noise reduction (faster)
```

**Output:** `output/cleaned/STEM_0000.wav`, `STEM_0001.wav`, …

---

### Stage 2 — Audio Concatenation *(optional)*

Concatenates cleaned clips into longer files for review or alignment.

```bash
python stage2_concat.py \
  --input    ./output/cleaned \
  --output   ./output/concatenated \
  --config   config.yaml

# Optional:
#   --manifest order.txt   text file with clip filenames in desired order
#   --silence-ms 500       silence gap between clips (ms)
#   --max-duration 300     split output at this duration (seconds)
```

---

### Stage 3 — TTS Generation *(optional — skip if you have recordings)*

Generates audio from a text file using Microsoft Edge-TTS (free, no key).

```bash
# List available French voices
python stage3_generate.py --list-voices

# Generate audio
python stage3_generate.py \
  --input  texts.txt \
  --output ./output/generated \
  --config config.yaml \
  --voice  fr-FR-DeniseNeural
```

**Input format** (`texts.txt`): one sentence per line, or a CSV with a `text` column.

**Output:** `00001.wav`, `00002.wav`, …, `metadata.csv`

**French voices:**
| Name | Gender | Description |
|------|--------|-------------|
| `fr-FR-DeniseNeural` | Female | Standard, clear (default) |
| `fr-FR-HenriNeural` | Male | Standard, clear |
| `fr-FR-EloiseNeural` | Female | Younger tone |
| `fr-BE-CharlineNeural` | Female | Belgian French |

---

### Stage 4 — Dataset Preparation

Validates audio, transcribes with Whisper, normalises text, and outputs
**LJSpeech format** ready for Coqui TTS training.

```bash
python stage4_prepare.py \
  --input  ./output/cleaned \
  --output ./output/dataset \
  --config config.yaml

# If you have no metadata.csv (pure audio → Whisper transcribes all):
python stage4_prepare.py \
  --input       ./output/cleaned \
  --output      ./output/dataset \
  --no-metadata \
  --config      config.yaml

# Skip transcription (much faster, no WER check):
python stage4_prepare.py \
  --input        ./output/cleaned \
  --output       ./output/dataset \
  --no-transcribe \
  --config       config.yaml
```

**Output layout:**
```
output/dataset/
    wavs/             ← symlinked WAV files
    metadata.csv      ← stem|raw_text|normalized_text (LJSpeech format)
    train.txt
    val.txt
    test.txt
    stats.csv         ← per-file duration, RMS, SNR, WER, flagged
    rejected.txt
    dataset_summary.json
```

**WER flagging:** Files with WER > `mismatch_threshold` (default 0.35) are
flagged in `stats.csv` but **not removed** — review them manually.

---

### Stage 5 — Model Training

#### VITS *(recommended for GTX 1650 4 GB)*

Trains a VITS model from scratch. Works comfortably in 4 GB VRAM with
`mixed_precision: true` and `batch_size: 8`.

```bash
python stage5_train.py \
  --dataset ./output/dataset \
  --output  ./output/training \
  --model   vits \
  --config  config.yaml

# Resume from checkpoint:
python stage5_train.py \
  --dataset ./output/dataset \
  --output  ./output/training \
  --model   vits \
  --resume  ./output/training/best_model.pth
```

#### XTTS v2 *(requires ≥ 6 GB VRAM)*

> **Warning:** GTX 1650 has 4 GB VRAM. XTTS v2 requires ≥ 6 GB.
> The training will likely run out of memory on your current hardware.
> Options:
> - Use **VITS** (above) — excellent quality for French TTS
> - Rent a cloud GPU: [RunPod](https://runpod.io), [Vast.ai](https://vast.ai), Google Colab A100

```bash
# Check VRAM before running XTTS:
python stage5_train.py --check-hardware --model xtts

# Fine-tune XTTS v2 (needs 6+ GB VRAM):
python stage5_train.py \
  --dataset ./output/dataset \
  --output  ./output/training \
  --model   xtts \
  --config  config.yaml
```

#### TensorBoard

```bash
tensorboard --logdir ./output/training
# then open http://localhost:6006
```

---

## End-to-End Pipeline

```bash
# Stages 1, 4, 5 (you have existing audio, skip Stage 3)
python run_pipeline.py --config config.yaml --stages 1 4 5

# All stages including TTS generation (Stage 3 needs --texts)
python run_pipeline.py \
  --config config.yaml \
  --stages 1 2 3 4 5 \
  --texts  texts.txt

# Skip transcription (fastest, no Whisper):
python run_pipeline.py \
  --config       config.yaml \
  --stages       1 4 5 \
  --no-transcribe

# Dry run (print commands, don't execute):
python run_pipeline.py --config config.yaml --stages 1 4 5 --dry-run
```

---

## French Medical Text Normalisation

`utils/text_utils.py` handles 40+ French medical abbreviations:

| Input | Normalised |
|-------|-----------|
| `FC 90 bpm` | `fréquence cardiaque 90 battements par minute` |
| `TA 120/80 mmHg` | `tension artérielle 120/80 millimètres de mercure` |
| `Dx: HTA, DT2` | `diagnostique: hypertension artérielle, diabète de type deux` |
| `Traitement IV 5mg/kg` | `traitement intraveineuse 5 milligrammes par kilogramme` |
| `ECG normal` | `électrocardiogramme normal` |

Add custom abbreviations in `config.yaml` (not yet exposed) or directly
in `utils/text_utils.py` → `MEDICAL_ABBREVIATIONS`.

---

## Configuration Reference

See `config.yaml` for all parameters. Key ones:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | `22050` | Audio sample rate for all stages |
| `stage1.normalization` | `rms` | `rms` or `peak` |
| `stage1.noise_reduction.enabled` | `true` | Apply noisereduce |
| `stage1.vad.method` | `silero` | `silero` (better) or `librosa` (faster) |
| `stage1.segment_filter.min_duration_s` | `1.0` | Minimum clip duration |
| `stage1.segment_filter.max_duration_s` | `15.0` | Maximum clip duration |
| `stage3.voice` | `fr-FR-DeniseNeural` | Edge-TTS voice |
| `stage4.whisper_model` | `medium` | `tiny/base/small/medium/large` |
| `stage4.mismatch_threshold` | `0.35` | WER flag threshold |
| `stage4.train_ratio` | `0.90` | Train split ratio |
| `stage5.model` | `vits` | `vits` or `xtts` |
| `stage5.vits.batch_size` | `8` | Reduce to `4` if OOM |
| `stage5.vits.mixed_precision` | `true` | fp16 training |

---

## Hardware Recommendations

| Model | Min VRAM | GTX 1650 (4 GB) |
|-------|----------|-----------------|
| VITS  | 3–4 GB   | ✅ Works         |
| YourTTS | 4–6 GB | ⚠️ Tight         |
| XTTS v2 | 6–8 GB | ❌ OOM           |

For XTTS v2 training, rent a GPU:
- **RunPod**: RTX 3090 (~$0.44/hr), A100 (~$1.89/hr)
- **Vast.ai**: similar pricing, more options
- **Google Colab Pro+**: A100 (40 GB)

---

## Troubleshooting

**`silero_vad` not found** → Stage 1 falls back to librosa VAD automatically.

**`espeak-ng` not found** → VITS phonemizer fails. Install espeak-ng and ensure it's on PATH.

**CUDA OOM during training** → Reduce `batch_size` in config.yaml, or set `mixed_precision: true`.

**Whisper CUDA OOM** → Set `stage4.whisper_device: cpu` in config.yaml.

**Edge-TTS network error** → Check your internet connection; edge-tts requires HTTPS access to Microsoft servers.

**WER > 0.35 on many files** → Your audio may have strong background noise or heavy accent. Lower the threshold or re-run Stage 1 with stronger noise reduction.
