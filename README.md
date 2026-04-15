# Automatic Music Transcription (AMT)

An end-to-end deep learning pipeline for automatic piano transcription using the MAESTRO v3 dataset. Converts raw piano audio into mel spectrograms, trains models to predict 88-key piano rolls, and evaluates three approaches: a traditional signal processing baseline (pYIN), a CNN + BiLSTM, and a CNN + Transformer.

---

## Technical Paper - Automatic Piano Transcription: A Comparison of Traditional Signal Processing and CNN-Based Deep Learning Models

Check the files in repository for PDF version of our technical paper

## Results Summary

| Model | Parameters | Precision | Recall | F1 (mean ± std) | F1 (median) |
|---|---|---|---|---|---|
| Traditional SP (pYIN) | 0 | 0.563 | 0.027 | 0.050 ± 0.040 | 0.035 |
| CNN + BiLSTM | 3.1M | 0.691 | 0.902 | 0.781 ± 0.039 | 0.786 |
| CNN + Transformer | 2.0M | **0.750** | **0.899** | **0.816 ± 0.037** | **0.822** |

Evaluated on 26 held-out validation songs (file-level split, seed=42).

---

## Dataset — MAESTRO v3

We use the [MAESTRO v3](https://magenta.tensorflow.org/datasets/maestro) dataset (MIDI + aligned audio) as the ground-truth target for AMT.

- ~200 hours of paired `.wav` and `.midi` recordings from ten years of the International Piano-e-Competition, recorded on Yamaha Disklavier pianos with precise alignment.
- Full dataset is ~120 GB. We use the **2015 subset (129 files)** during development to verify the pipeline before scaling up.
- Place raw files under `Audio Files/2015/` — all preprocessing scripts expect audio and MIDI in that folder by default.

### Piano Roll Format

- 88 rows corresponding to MIDI pitches 21–108 (A0–C8), the standard piano range.
- Binary values per time frame: `1` if a note is active, `0` otherwise.
- Time frames are aligned to mel spectrogram frames via STFT hop length. Piano rolls are trimmed or padded to exactly match mel frame counts.
- Average polyphony: **4.6 simultaneous notes per frame**; note density: **~5%** (95% of frame-key pairs are silent).

---

## Project Structure

```
.
├── Audio Files/
│   └── 2015/                    # Raw .wav and .midi files (MAESTRO 2015 subset)
├── mels/                        # Mel spectrogram .npy arrays (+ optional PNG previews)
├── pianorolls/                  # 88-key piano roll .npy arrays
│   └── pairs.csv                # Links each audio file to its mel and piano roll
├── predicted_midis/             # MIDI files output by the traditional pipeline
├── EDA/
│   ├── visualizations/          # Paired mel + piano roll sample plots (6 files)
│   ├── pitch_hist.png           # Pitch activation histogram across dataset
│   └── dataset_analysis.md     # Full written EDA summary
├── outputs/
│   ├── checkpoints/             # Saved model weights + TensorBoard logs
│   │   ├── best_cnn_transformer.pt
│   │   ├── best_cnn_bilstm.pt
│   │   ├── latest_*.pt
│   │   └── logs/                # TensorBoard event files
│   └── transcriptions/          # Per-file inference outputs (PNG + optional MIDI)
├── scripts/
│   ├── convert_to_mel.py        # Audio → mel spectrogram (GPU via torchaudio)
│   ├── midi_to_pianoroll.py     # MIDI → 88-key piano roll + pairs.csv
│   ├── eda_visualize.py         # Exploratory data analysis + statistics
│   ├── dataset.py               # MusicTranscriptionDataset + get_dataloaders()
│   ├── model.py                 # Model definitions (TraditionalSP, CNNBiLSTM, CNNTransformer)
│   ├── train.py                 # Training loop with checkpointing + TensorBoard
│   ├── evaluate.py              # Batch evaluation on val split — prints P/R/F1 table
│   └── inference.py             # Transcribe any audio file → piano roll PNG / MIDI
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124
```

> The `--index-url` flag pulls CUDA-enabled builds of torch and torchaudio. For CPU-only machines replace `cu124` with `cpu`.

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

> `torch` and `torchaudio` must be the **same version**. Mismatched versions cause a `WinError 127` / `OSError` at import time. Always install them together via the PyTorch index URL above.

---

## Full Pipeline

### Step 1 — Audio → Mel Spectrograms

```bash
python scripts/convert_to_mel.py \
    --input-dir "Audio Files/2015" \
    --output-dir mels \
    --save-png
```

Loads each audio file using torchaudio (GPU-accelerated), computes a log-power mel spectrogram, and saves a `.npy` array per file under `mels/`, mirroring the input folder structure. `--save-png` also saves a visual preview alongside each array.

**Output shape:** `(128, T)` — 128 mel bins × T time frames.

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | `Audio Files/2015` | Folder to scan recursively |
| `--output-dir` | `mels` | Where to write `.npy` files |
| `--sr` | 22050 | Sample rate (Hz) |
| `--n-mels` | 128 | Mel frequency bins |
| `--n-fft` | 2048 | FFT window size |
| `--hop-length` | 512 | Hop length (determines frame rate) |
| `--fmax` | 8000 | Max frequency (Hz) |
| `--save-png` | off | Also save PNG preview |
| `--exts` | .wav .mp3 .flac | Audio extensions to scan |

---

### Step 2 — MIDI → Piano Rolls

```bash
python scripts/midi_to_pianoroll.py \
    --midi-dir "Audio Files/2015" \
    --mel-dir mels \
    --out-dir pianorolls
```

Converts each `.midi` file to an 88-key binary piano roll at the same time grid as the mel (hop_length=512). If the corresponding mel exists, the piano roll is trimmed or padded to exactly match its frame count. Writes `pianorolls/pairs.csv` which links every file triplet `(midi, piano_roll, mel)` — this CSV drives all subsequent steps.

**Output shape:** `(88, T)` — 88 piano keys × T time frames (binary).

| Flag | Default | Description |
|------|---------|-------------|
| `--midi-dir` | `Audio Files/2015` | Folder containing MIDI files |
| `--mel-dir` | `mels` | Folder with precomputed mel `.npy` files |
| `--out-dir` | `pianorolls` | Where to write piano roll `.npy` files |
| `--sr` | 22050 | Must match mel preprocessing |
| `--hop-length` | 512 | Must match mel preprocessing |
| `--save-npz` | off | Also save paired mel+roll as `.npz` |

---

### Step 3 — Exploratory Data Analysis (optional)

```bash
python scripts/eda_visualize.py \
    --pairs-csv pianorolls/pairs.csv \
    --out-dir EDA \
    --num-samples 6
```

Generates paired mel + piano roll visualizations for 6 sample files, a pitch activation histogram across the full dataset, and a written Markdown summary saved to `EDA/dataset_analysis.md`.

Key findings from the 2015 subset:
- 129 files → 2.2 million total frames
- Average polyphony: 4.6 notes/frame
- Note density: 5.06% (severe class imbalance — addressed in training via `pos_weight`)

---

### Step 4 — Training

All three models share the same training script. Swap `--model-type` to choose the architecture.

```bash
# CNN + Transformer (best model)
python scripts/train.py \
    --pairs-csv pianorolls/pairs.csv \
    --mels-dir mels \
    --pianorolls-dir pianorolls \
    --model-type cnn_transformer \
    --epochs 50 --batch-size 16 --lr 0.001 \
    --output-dir outputs/checkpoints

# CNN + BiLSTM
python scripts/train.py \
    --pairs-csv pianorolls/pairs.csv \
    --mels-dir mels \
    --pianorolls-dir pianorolls \
    --model-type cnn_bilstm \
    --epochs 30 --batch-size 16 --lr 0.001 \
    --output-dir outputs/checkpoints

# Resume / fine-tune from a checkpoint
python scripts/train.py \
    --pairs-csv pianorolls/pairs.csv \
    --mels-dir mels \
    --pianorolls-dir pianorolls \
    --model-type cnn_transformer \
    --resume outputs/checkpoints/best_cnn_transformer.pt \
    --lr 0.0005 --epochs 20 \
    --output-dir outputs/checkpoints
```

**What happens internally:**
- Files are split 80/20 at the **file level** (103 train / 26 val) before chunking — no song leaks between splits
- Each file is cut into non-overlapping 256-frame chunks (~6 sec each) → 8,623 total chunks
- Loss: `BCEWithLogitsLoss` with `pos_weight=19` to compensate for the 5% note density imbalance
- Optimizer: Adam with `ReduceLROnPlateau` (patience=3, factor=0.5)
- Gradient clipping: `max_norm=1.0`
- Best checkpoint saved by **validation F1** (not loss)

**Outputs:**
```
outputs/checkpoints/
├── best_cnn_transformer.pt    ← best checkpoint by val F1
├── latest_cnn_transformer.pt  ← most recent epoch
└── logs/                      ← TensorBoard event files
```

**Monitor training live:**
```bash
tensorboard --logdir outputs/checkpoints/logs
```

---

### Step 5 — Evaluation

Reproduces the exact val split from training (seed=42, 80/20 file split) and reports frame-level P/R/F1 across all 26 held-out files.

```bash
# Traditional baseline — no checkpoint needed
python scripts/evaluate.py \
    --pairs-csv pianorolls/pairs.csv \
    --model-type traditional

# CNN + Transformer
python scripts/evaluate.py \
    --pairs-csv pianorolls/pairs.csv \
    --model-type cnn_transformer \
    --model-path outputs/checkpoints/best_cnn_transformer.pt

# CNN + BiLSTM
python scripts/evaluate.py \
    --pairs-csv pianorolls/pairs.csv \
    --model-type cnn_bilstm \
    --model-path outputs/checkpoints/best_cnn_bilstm.pt

# Evaluate on ALL files (not just val split)
python scripts/evaluate.py \
    --pairs-csv pianorolls/pairs.csv \
    --model-type cnn_transformer \
    --model-path outputs/checkpoints/best_cnn_transformer.pt \
    --all-files
```

**Example output:**
```
====================================================
  Method     : cnn_transformer
  Files      : 26 evaluated, 0 skipped
  ────────────────────────────────────────────────
  Precision  : 0.7500
  Recall     : 0.8990
  F1 (mean)  : 0.8160  ±0.0370
  F1 (median): 0.8220
====================================================
```

---

### Step 6 — Inference

Transcribe any audio file using a saved checkpoint. Outputs a piano roll PNG and optionally a `.mid` file.

```bash
# Predicted piano roll only
python scripts/inference.py \
    --audio "Audio Files/2015/some_piece.wav" \
    --model-type cnn_transformer \
    --model-path outputs/checkpoints/best_cnn_transformer.pt \
    --output-dir outputs/transcriptions

# Ground truth vs predicted comparison + MIDI export
python scripts/inference.py \
    --audio "Audio Files/2015/some_piece.wav" \
    --model-type cnn_transformer \
    --model-path outputs/checkpoints/best_cnn_transformer.pt \
    --midi-path "Audio Files/2015/some_piece.midi" \
    --save-midi \
    --output-dir outputs/transcriptions

# Traditional baseline — no checkpoint needed
python scripts/inference.py \
    --audio "Audio Files/2015/some_piece.wav" \
    --model-type traditional
```

**Outputs in** `outputs/transcriptions/`:
```
some_piece_cnn_transformer_piano_roll.png   ← predicted roll (blue)
some_piece_cnn_transformer_comparison.png  ← GT (green, top) vs predicted (blue, bottom)
                                               with P/R/F1 in the title
some_piece_cnn_transformer.mid             ← exportable MIDI file
```

| Flag | Default | Description |
|------|---------|-------------|
| `--audio` | required | Path to input audio file |
| `--model-type` | `cnn_transformer` | `traditional`, `cnn_bilstm`, or `cnn_transformer` |
| `--model-path` | — | Checkpoint `.pt` (required for CNN models) |
| `--threshold` | 0.5 | Lower = more notes, more false positives |
| `--midi-path` | — | Ground truth MIDI for comparison image |
| `--save-midi` | off | Export a `.mid` file |
| `--output-dir` | `outputs/transcriptions` | Where to write outputs |

---

## Model Architectures

### Traditional SP (pYIN) — 0 parameters
Probabilistic YIN pitch detection + onset-based note segmentation. Fully interpretable, no training needed. **Core limitation:** pYIN tracks one pitch per frame — piano music averages 4.6 simultaneous notes, so ~97% of active notes are missed (Recall = 0.027).

### CNN + BiLSTM — 3.1M parameters
Two Conv1d layers (128→256 channels, BatchNorm + Dropout) feed into a 2-layer Bidirectional LSTM (hidden=256 per direction). The BiLSTM propagates state across the full 256-frame chunk; bidirectionality gives every frame access to both past and future context, capturing the full attack → sustain → release lifecycle of each note.

### CNN + Transformer — 2.0M parameters
Same CNN encoder feeds into a 2-layer Transformer encoder (8-head self-attention, 1024-dim feedforward, sinusoidal positional encoding). Self-attention connects any two frames directly in one operation, learning long-range harmonic dependencies. Achieves higher precision than BiLSTM (+0.059) with fewer parameters — global attention suppresses false positives caused by harmonic interference more effectively than sequential LSTM gates.

---

## Data Flow

```
Audio Files/2015/
  ├── piece.wav  ──► convert_to_mel.py ──► mels/piece.npy          (128 × T)
  └── piece.midi ──► midi_to_pianoroll.py ──► pianorolls/piece.npy  (88 × T)
                                          └──► pianorolls/pairs.csv

pairs.csv ──► train.py ──► outputs/checkpoints/best_*.pt
                       └──► outputs/checkpoints/logs/   (TensorBoard)

best_*.pt + any audio ──► inference.py ──► outputs/transcriptions/
                                            ├── *_comparison.png
                                            └── *.mid
```

---

## Common Issues

**`WinError 127` / `OSError` when importing torchaudio**
Version mismatch between `torch` and `torchaudio`. Fix:
```bash
pip uninstall torch torchaudio -y
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

**`No audio files found`**
Make sure files are under `Audio Files/2015/`. The space in the folder name requires quoting the path in the shell: `--input-dir "Audio Files/2015"`.

**Training F1 stuck near zero**
The 5% note density means the model defaults to predicting silence. Ensure `--pos-weight` is set (default 19 ≈ `(1 - 0.05) / 0.05`). Adjust proportionally if using a different subset with different density.

**`--model-path is required` error**
The traditional baseline needs no checkpoint, but `cnn_bilstm` and `cnn_transformer` always require `--model-path` pointing to a `.pt` file.
