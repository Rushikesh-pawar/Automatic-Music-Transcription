"""
Batch evaluation script: compute frame-level P / R / F1 across the validation
set for any transcription method and print a summary table.

The validation split uses the same file-level seed as train.py (seed=42,
last 20% of shuffled files) so results are comparable to training metrics.

Usage
-----
# Traditional SP (no checkpoint):
python scripts/evaluate.py --pairs-csv pianorolls/pairs.csv \
                            --model-type traditional

# Learned model:
python scripts/evaluate.py --pairs-csv pianorolls/pairs.csv \
                            --model-type cnn_transformer \
                            --model-path outputs/checkpoints/best_cnn_transformer.pt

# Evaluate on ALL files instead of just the val split:
python scripts/evaluate.py --pairs-csv pianorolls/pairs.csv \
                            --model-type traditional --all-files
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from model import get_model

# Must match preprocessing / training constants
SR         = 22050
N_FFT      = 2048
HOP_LENGTH = 512
FMAX       = 8000
CHUNK_SIZE = 256
PIANO_LOW  = 21


# ---------------------------------------------------------------------------
# Helpers shared with inference.py
# ---------------------------------------------------------------------------

def normalize(mel: np.ndarray) -> np.ndarray:
    lo, hi = mel.min(), mel.max()
    return (mel - lo) / (hi - lo + 1e-8)


def transcribe_mel(model, mel: np.ndarray, device: torch.device,
                   threshold: float = 0.5) -> np.ndarray:
    """Run a learned model on a full mel and return binary piano roll (88, T)."""
    model.eval()
    T = mel.shape[-1]
    probs_full = np.zeros((88, T), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, T, CHUNK_SIZE):
            end       = min(start + CHUNK_SIZE, T)
            chunk     = mel[:, start:end].copy()
            chunk_len = chunk.shape[-1]
            if chunk_len < CHUNK_SIZE:
                chunk = np.pad(chunk, ((0, 0), (0, CHUNK_SIZE - chunk_len)))
            x = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(device)
            _, p = model(x)
            probs_full[:, start:end] = p[0].cpu().numpy()[:, :chunk_len]

    return (probs_full >= threshold).astype(np.float32)


def transcribe_traditional(audio_path: str, n_frames: int,
                            threshold: float = 0.5) -> np.ndarray:
    """
    pYIN-based transcription. Returns binary piano roll (88, n_frames).
    Matches the implementation in inference.py.
    """
    y, _ = librosa.load(audio_path, sr=SR, mono=True)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('A0'),
        fmax=librosa.note_to_hz('C8'),
        sr=SR,
        hop_length=HOP_LENGTH,
        frame_length=N_FFT,
    )

    onsets    = librosa.onset.onset_detect(y=y, sr=SR, hop_length=HOP_LENGTH,
                                           units='frames')
    onset_set = set(onsets.tolist())

    probs             = np.zeros((88, n_frames), dtype=np.float32)
    T                 = min(len(f0), n_frames)
    current_pitch_idx = None

    for t in range(T):
        if voiced_flag[t] and not np.isnan(f0[t]):
            midi_pitch = int(round(librosa.hz_to_midi(f0[t])))
            if PIANO_LOW <= midi_pitch < PIANO_LOW + 88:
                key_idx = midi_pitch - PIANO_LOW
                probs[key_idx, t]  = float(voiced_prob[t])
                current_pitch_idx  = key_idx
        elif current_pitch_idx is not None and t not in onset_set:
            probs[current_pitch_idx, t] = 0.3
        else:
            current_pitch_idx = None

    return (probs >= threshold).astype(np.float32)


def find_audio(midi_path: str) -> str | None:
    """
    Find the audio file that corresponds to a MIDI file.
    Tries common audio extensions in the same directory.
    """
    base = os.path.splitext(midi_path)[0]
    for ext in ('.wav', '.mp3', '.flac', '.ogg', '.m4a'):
        candidate = base + ext
        if os.path.isfile(candidate):
            return candidate
    return None


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    tp   = (pred * target).sum()
    fp   = (pred * (1 - target)).sum()
    fn   = ((1 - pred) * target).sum()
    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    f1   = 2 * prec * rec / (prec + rec + 1e-6)
    return {'precision': float(prec), 'recall': float(rec), 'f1': float(f1)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate transcription accuracy on the validation set')
    parser.add_argument('--pairs-csv', required=True,
                        help='Path to pairs.csv (same as used for training)')
    parser.add_argument('--model-type', required=True,
                        choices=['traditional', 'cnn_bilstm', 'cnn_transformer'])
    parser.add_argument('--model-path', default=None,
                        help='Checkpoint .pt file (required for CNN models)')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Must match the value used during training (default 0.8)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--all-files', action='store_true',
                        help='Evaluate on all files, not just the val split')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if args.model_type != 'traditional' and args.model_path is None:
        sys.exit(f"--model-path is required for --model-type {args.model_type}")

    # ---- Reproduce the same file-level val split as train.py ----
    all_files = pd.read_csv(args.pairs_csv)
    all_files = all_files.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    if args.all_files:
        val_df = all_files
        print(f"Evaluating on ALL {len(val_df)} files")
    else:
        n_train = int(args.train_split * len(all_files))
        val_df  = all_files.iloc[n_train:]
        print(f"Evaluating on {len(val_df)} validation files "
              f"(held out from training, seed={args.seed})")

    # ---- Load model (CNN only) ----
    device = torch.device(args.device)
    model  = None
    if args.model_type != 'traditional':
        model = get_model(args.model_type).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        print(f"Loaded checkpoint: {args.model_path}")

    print(f"Method    : {args.model_type}")
    print(f"Threshold : {args.threshold}\n")

    # ---- Evaluate file by file ----
    results = []
    skipped = 0

    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc='Evaluating'):
        pr_path  = row['piano_roll']
        mel_path = row['mel']

        # Load ground-truth piano roll
        if not os.path.isfile(pr_path):
            skipped += 1
            continue
        target = np.load(pr_path).astype(np.float32)  # (88, T)
        n_frames = target.shape[-1]

        try:
            if args.model_type == 'traditional':
                midi_path  = row['midi']
                audio_path = find_audio(midi_path)
                if audio_path is None:
                    tqdm.write(f"  [skip] no audio found for: {midi_path}")
                    skipped += 1
                    continue
                pred = transcribe_traditional(audio_path, n_frames, args.threshold)

            else:
                if not os.path.isfile(mel_path):
                    skipped += 1
                    continue
                mel  = normalize(np.load(mel_path).astype(np.float32))
                pred = transcribe_mel(model, mel, device, args.threshold)

            # Align lengths (pred may differ by 1 frame due to padding)
            T = min(pred.shape[-1], target.shape[-1])
            metrics = compute_metrics(pred[:, :T], target[:, :T])
            results.append(metrics)

        except Exception as exc:
            tqdm.write(f"  [error] {os.path.basename(pr_path)}: {exc}")
            skipped += 1

    # ---- Aggregate and print ----
    if not results:
        print("No files evaluated successfully.")
        return

    prec_mean = np.mean([r['precision'] for r in results])
    rec_mean  = np.mean([r['recall']    for r in results])
    f1_mean   = np.mean([r['f1']        for r in results])
    f1_std    = np.std( [r['f1']        for r in results])
    f1_med    = np.median([r['f1']      for r in results])

    print(f"\n{'='*52}")
    print(f"  Method     : {args.model_type}")
    print(f"  Files      : {len(results)} evaluated, {skipped} skipped")
    print(f"{'─'*52}")
    print(f"  Precision  : {prec_mean:.4f}")
    print(f"  Recall     : {rec_mean:.4f}")
    print(f"  F1 (mean)  : {f1_mean:.4f}  ±{f1_std:.4f}")
    print(f"  F1 (median): {f1_med:.4f}")
    print(f"{'='*52}")


if __name__ == '__main__':
    main()
