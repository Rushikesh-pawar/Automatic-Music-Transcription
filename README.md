# Automatic Music Transcription (AMT)

A CNN-based pipeline for automatic piano transcription using the MAESTRO v3 dataset. Converts raw audio into mel spectrograms, trains models to predict 88-key piano rolls, and evaluates three approaches: a traditional signal processing baseline, a CNN + BiLSTM, and a CNN + Transformer.

---

## Dataset — MAESTRO v3

We use the [MAESTRO v3](https://magenta.tensorflow.org/datasets/maestro) dataset (MIDI + aligned audio) as the ground-truth target for AMT.

- Contains ~200 hours of paired `.wav` and `.midi` recordings from ten years of the International Piano-e-Competition, recorded on Yamaha Disklavier pianos with precise alignment.
- Full dataset is ~120 GB. We use the 2015 subset (129 files) during development to verify the pipeline before scaling up.
- Place raw files under `Audio Files/2015/` — all preprocessing scripts expect audio and MIDI in that folder by default.
- We use the audio to create mel spectrogram inputs and the MIDI to create 88-key piano roll targets.

### Piano Roll Format

- 88 rows corresponding to MIDI pitches 21–108 (A0–C8), the standard piano range.
- Binary values per time frame: `1` if a note is active, `0` otherwise.
- Time frames are aligned to mel spectrogram frames — the STFT hop length determines the frame rate. Piano rolls are trimmed or padded to exactly match mel frame counts.

---

## Project Structure

```
.
├── Audio Files/
│   └── 2015/               # Raw .wav and .midi files (MAESTRO 2015 subset)
├── mels/                   # Mel spectrogram .npy arrays (+ optional PNG previews)
├── pianorolls/             # 88-key piano roll .npy arrays
│   └── pairs.csv           # Links each audio file to its mel and piano roll
├── EDA/
│   ├── visualizations/     # Paired mel + piano roll sample plots
│   ├── pitch_hist.png      # Pitch activation histogram
│   └── dataset_analysis.md # Full written EDA summary
├── scripts/
│   ├── convert_to_mel.py   # Audio → mel spectrogram
│   ├── midi_to_pianoroll.py# MIDI → piano roll
│   ├── eda_visualize.py    # Exploratory data analysis
│   ├── dataset.py          # Dataset class + dataloaders
│   ├── model.py            # Model definitions (3 architectures)
│   ├── train.py            # Training loop
│   └── inference.py        # Transcribe any audio file
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124
```

> The `--index-url` flag pulls CUDA-enabled builds of torch and torchaudio. Omit it or use `/cpu` for CPU-only installs.

**requirements.txt**
```
librosa
soundfile
numpy
matplotlib
tqdm
pretty_midi
torch==2.6.0
torchaudio==2.6.0
torchvision
pandas
tensorboard
```

---

## Preprocessing

### 1. Audio → Mel Spectrograms

```bash
python scripts/convert_to_mel.py --input-dir "Audio Files/2015" --output-dir mels --save-png
```

Produces `.npy` mel arrays (and optional PNG previews) under `mels/`, mirroring the input folder structure.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sr` | 22050 | Sample rate (Hz) |
| `--n-mels` | 128 | Number of mel bins |
| `--n-fft` | 2048 | FFT window size |
| `--hop-length` | 512 | Hop length (frames) |
| `--fmax` | 8000 | Max frequency (Hz) |
| `--save-png` | off | Also save PNG previews |
| `--exts` | .wav .mp3 .flac | Audio extensions to scan |

### 2. MIDI → Piano Rolls

```bash
python scripts/midi_to_pianoroll.py --input-dir "Audio Files/2015" --output-dir pianorolls
```

Produces 88-key binary piano roll `.npy` arrays aligned to the mel frame grid, and writes `pianorolls/pairs.csv` linking each audio file to its mel and piano roll.

---

## Training

```bash
# Train CNN + Transformer (best model)
python scripts/train.py --model-type cnn_transformer --output-dir runs/transformer

# Train CNN + BiLSTM
python scripts/train.py --model-type cnn_bilstm --output-dir runs/bilstm

# Resume from checkpoint
python scripts/train.py --model-type cnn_transformer --resume runs/transformer/best.pt
```

Training uses Adam with `ReduceLROnPlateau` (patience=3, factor=0.5), gradient clipping (max_norm=1.0), and best-model checkpointing on validation F1. Monitor with TensorBoard:

```bash
tensorboard --logdir runs/
```

---

## Inference

Transcribe any audio file using a saved checkpoint:

```bash
python scripts/inference.py --audio path/to/audio.wav --checkpoint runs/transformer/best.pt --save-midi
```

Add `--midi-path` to produce a stacked ground truth vs. predicted comparison image with frame-level P/R/F1 in the title:

```bash
python scripts/inference.py --audio path/to/audio.wav --checkpoint runs/transformer/best.pt \
    --midi-path path/to/ground_truth.midi --save-midi
```

---

## Results

Evaluated on 26 held-out validation songs (file-level split to prevent data leakage).

| Model | Parameters | Precision | Recall | F1 (mean ± std) | F1 (median) |
|---|---|---|---|---|---|
| Traditional SP (pYIN) | 0 | 0.563 | 0.027 | 0.050 ± 0.040 | 0.035 |
| CNN + BiLSTM | 3.1M | 0.691 | 0.902 | 0.781 ± 0.039 | 0.786 |
| CNN + Transformer | 2.0M | **0.750** | **0.899** | **0.816 ± 0.037** | **0.822** |

The 15× F1 jump from pYIN to CNN models comes from the ability to detect multiple simultaneous pitches — something monophonic pitch detection cannot do. The Transformer's edge over the BiLSTM (+0.035 F1, +0.059 precision) reflects the benefit of global self-attention over sequential hidden-state propagation for modeling harmonic context across the full 256-frame window.
