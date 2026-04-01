# Automatic Music Transcription — Progress Report


**Project:** Automatic Music Transcription (AMT)

---

## What We Have Completed

### 1. Dataset and Preprocessing Pipeline

We use the **MAESTRO v3** dataset (professional piano recordings with aligned MIDI ground truth). From the 2015 subset we extracted **129 paired audio/MIDI recordings**.

Two preprocessing scripts were built and run in full:

- **`scripts/convert_to_mel.py`** — converts each `.wav` file to a log-power mel spectrogram saved as a NumPy array (shape: 128 mel bins × T frames). Parameters chosen to match the piano frequency range: `sr=22050`, `n_fft=2048`, `hop_length=512`, `n_mels=128`, `fmax=8000 Hz`.
- **`scripts/midi_to_pianoroll.py`** — converts each `.midi` file to an 88-key binary piano roll aligned to the same time grid as the mel spectrogram (shape: 88 keys × T frames). Outputs a `pairs.csv` linking each audio file to its mel and piano-roll arrays.

All 129 files were successfully processed, producing **8,623 fixed-length chunks** of 256 frames each (≈ 6 seconds per chunk).

### 2. Exploratory Data Analysis

**`scripts/eda_visualize.py`** generates:
- Paired mel spectrogram + piano roll visualisations for 6 sample files (saved in `EDA/visualizations/`)
- Dataset statistics: average polyphony of **4.6 simultaneous notes per frame**, note density of **5.06%** (sparse), 2.2 million total frames across all files
- Pitch activation histogram (saved as `EDA/pitch_hist.png`)
- Full written summary in `EDA/dataset_analysis.md`

Key insight from EDA: the 5% note density revealed a severe class imbalance (95% of frame-key pairs are silent), which directly informed our loss function design.

### 3. Data Loading with Proper Train/Validation Split

**`scripts/dataset.py`** implements a `MusicTranscriptionDataset` class and `get_dataloaders()` function with the following design decisions:

- **Chunk-based indexing**: all non-overlapping 256-frame chunks are pre-indexed at init time so every frame is seen each epoch (8,623 chunks vs. 129 files).
- **File-level split**: train/validation split is performed at the *file* level (103 train / 26 val files) before chunks are created, preventing the same song from appearing in both sets (data leakage prevention).
- **Class imbalance handling**: `BCEWithLogitsLoss` with `pos_weight=10` upweights the rare positive (note-active) class, addressing the 5% note density imbalance.

### 4. Three-Model Comparison

**`scripts/model.py`** defines three approaches to frame-level transcription, all sharing the same interface (`forward` returns `logits, probs` of shape `(batch, 88, time)`):

#### Model 1: Traditional Signal Processing (`traditional`)
A parameter-free baseline using classical audio analysis. The approach combines **Harmonic Product Spectrum (HPS)** — which reinforces fundamental frequencies by multiplying harmonically downsampled copies of the spectrum — with per-key energy thresholding. This method requires no training data and is fully interpretable. A stub implementation is in place; we are working on the complete signal-processing pipeline and will be integrated before final submission.

#### Model 2: CNN + BiLSTM (`cnn_bilstm`) — 3.1M parameters
A CNN encoder (two Conv1d layers with BatchNorm and dropout) feeds into a **2-layer Bidirectional LSTM** with 256 hidden units per direction. The BiLSTM propagates hidden state across the entire chunk, allowing it to model the full temporal lifecycle of a note (attack → sustain → release). Bidirectionality gives each frame access to both past and future context.

**Best validation F1: 0.7753** (file-level split, 26 unseen songs)

#### Model 3: CNN + Transformer (`cnn_transformer`) — 2.0M parameters
The same CNN encoder feeds into a **2-layer Transformer encoder** (8-head self-attention, 1024-dim feedforward, sinusoidal positional encoding). Self-attention directly connects any two frames in a single operation, learning long-range harmonic dependencies that the LSTM must propagate through hundreds of gate operations. It also uses fewer parameters than the BiLSTM.

**Best validation F1: 0.8160** (after 50 total epochs including fine-tuning at lr=0.0005)

| Model | Params | Best F1 | Precision | Recall |
|---|---|---|---|---|
| Traditional SP | 0 | (pending) | — | — |
| CNN + BiLSTM | 3.1M | 0.7753 | 0.68 | 0.90 |
| CNN + Transformer | 2.0M | **0.8160** | **0.75** | 0.89 |

### 5. Training Infrastructure

**`scripts/train.py`** provides a complete training loop with:
- Adam optimiser, `ReduceLROnPlateau` learning rate scheduler (patience=3, factor=0.5)
- Gradient clipping (`max_norm=1.0`) to stabilise training
- Frame-level Precision, Recall, and F1 metrics computed each epoch
- TensorBoard logging of all losses and metrics
- Best-model checkpointing based on validation F1 (not loss)
- `--resume` flag for fine-tuning from a saved checkpoint
- `--model-type` flag to select any of the three models without changing code

### 6. Inference Pipeline

**`scripts/inference.py`** transcribes any audio file (.wav, .mp3, .flac) using any saved checkpoint:
- Loads audio → mel spectrogram using the same parameters as training
- Runs the model in 256-frame windows across the full recording
- Saves a piano roll visualisation as PNG
- When `--midi-path` is provided, produces a **stacked comparison image** (ground truth in green on top, predicted in blue below) with frame-level P/R/F1 printed in the title, enabling direct visual evaluation
- Optionally exports a `.mid` MIDI file using `pretty_midi`

---

## What We Plan to Do Before Final Submission

1. **Integrate the traditional signal processing pipeline** — we are working on the NMF-based and spectral peak-picking implementations. Once received, we will replace the stub in `TraditionalSP._sp_transcribe()` and run it on the full validation set to produce comparable F1 scores.

2. **Final three-way comparison** — run all three models on the held-out 26 validation songs, produce a unified comparison table (F1, Precision, Recall, inference speed), and generate side-by-side comparison PNG images using the inference script.

3. **Error analysis** — identify which note ranges, polyphony levels, and musical contexts each model struggles with. We expect the traditional method to perform worse on dense chords, and the CNN models to struggle more on very sparse passages.

4. **Continue CNN + Transformer training** — the model was still improving at the end of training (F1=0.8160, epoch 50). We plan to run additional fine-tuning epochs to see how high it can go.

5. **Written final report** — compile the comparison results, discuss the trade-offs between traditional and learned approaches, and reflect on what the F1 gap between CNN + BiLSTM and CNN + Transformer tells us about the importance of global vs. sequential temporal context in music transcription.
