"""
Inference script: transcribe any audio file to a piano roll.

Usage:
    # Learned models (require a trained checkpoint):
    python scripts/inference.py --audio path/to/song.wav \
                                 --model-path outputs/checkpoints/best_cnn_transformer.pt \
                                 --model-type cnn_transformer

    # Traditional signal processing (no checkpoint needed):
    python scripts/inference.py --audio path/to/song.wav \
                                 --model-type traditional

Optional flags:
    --midi-path path/to/ground_truth.midi   # show predicted vs ground truth side-by-side
    --threshold 0.4                          # lower = more notes detected (default 0.5)
    --save-midi                              # export a .mid file alongside the PNG
    --output-dir outputs/transcriptions
"""

import argparse
import os
import sys

import numpy as np
import torch
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# allow running from project root or scripts/
sys.path.insert(0, os.path.dirname(__file__))
from model import get_model

try:
    import pretty_midi
    _HAS_MIDI = True
except ImportError:
    _HAS_MIDI = False

# --- Must match the parameters used during preprocessing ---
SR          = 22050
N_MELS      = 128
N_FFT       = 2048
HOP_LENGTH  = 512
FMAX        = 8000
CHUNK_SIZE  = 256   # frames per inference window (same as training)
PIANO_LOW   = 21    # MIDI pitch of A0 (lowest piano key)


# ---------------------------------------------------------------------------
# Audio → mel
# ---------------------------------------------------------------------------

def audio_to_mel(audio_path: str) -> np.ndarray:
    """Load audio and return log-power mel spectrogram (128, T)."""
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmax=FMAX
    )
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


def normalize(mel: np.ndarray) -> np.ndarray:
    lo, hi = mel.min(), mel.max()
    return (mel - lo) / (hi - lo + 1e-8)


# ---------------------------------------------------------------------------
# Inference — learned models (CNNBiLSTM / CNNTransformer)
# ---------------------------------------------------------------------------

def transcribe(model: torch.nn.Module,
               mel: np.ndarray,
               device: torch.device,
               threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model on the full mel in non-overlapping chunks.

    Returns
    -------
    piano_roll_binary : (88, T) float32  — thresholded 0/1
    piano_roll_probs  : (88, T) float32  — raw probabilities
    """
    model.eval()
    T = mel.shape[-1]
    probs_full = np.zeros((88, T), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, T, CHUNK_SIZE):
            end   = min(start + CHUNK_SIZE, T)
            chunk = mel[:, start:end].copy()
            chunk_len = chunk.shape[-1]

            # Pad last chunk if shorter than CHUNK_SIZE
            if chunk_len < CHUNK_SIZE:
                chunk = np.pad(chunk,
                               ((0, 0), (0, CHUNK_SIZE - chunk_len)),
                               mode='constant')

            # (1, 1, 128, CHUNK_SIZE) → model → (1, 88, CHUNK_SIZE)
            x = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(device)
            _, p = model(x)
            probs_full[:, start:end] = p[0].cpu().numpy()[:, :chunk_len]

    return (probs_full >= threshold).astype(np.float32), probs_full


# ---------------------------------------------------------------------------
# Inference — traditional signal processing (pYIN pitch detection)
# ---------------------------------------------------------------------------

def transcribe_traditional(audio_path: str,
                            n_frames: int,
                            threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Traditional signal processing transcription using pYIN pitch detection
    and onset-based note segmentation (teammate's approach).

    pYIN tracks a single F0 per frame, so it works best on melodic lines
    rather than dense chords — this is the key limitation vs learned models,
    and makes the comparison instructive for showing why CNNs help.

    The output uses the same time grid as the mel spectrogram (hop_length=512)
    so piano rolls are directly comparable to CNN model outputs.

    Returns
    -------
    piano_roll_binary : (88, n_frames) float32  — thresholded 0/1
    piano_roll_probs  : (88, n_frames) float32  — voicing probability at each key
    """
    print("  Running pYIN pitch detection (this may take a moment) …")
    y, _ = librosa.load(audio_path, sr=SR, mono=True)

    # pYIN: frame-level F0 + voicing probability on the same hop grid as mel
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('A0'),   # MIDI 21 — lowest piano key
        fmax=librosa.note_to_hz('C8'),   # MIDI 108 — highest piano key
        sr=SR,
        hop_length=HOP_LENGTH,
        frame_length=N_FFT,
    )

    # Onset detection — used to extend note activations across the full note
    # duration instead of only voiced frames (sustain after attack goes silent)
    onsets = librosa.onset.onset_detect(y=y, sr=SR, hop_length=HOP_LENGTH,
                                        units='frames')
    onset_set = set(onsets.tolist())

    # Build piano roll: for each frame, if pYIN is voiced assign its pitch
    probs = np.zeros((88, n_frames), dtype=np.float32)
    T = min(len(f0), n_frames)

    current_pitch_idx = None   # key index of the active note

    for t in range(T):
        if voiced_flag[t] and not np.isnan(f0[t]):
            midi_pitch = int(round(librosa.hz_to_midi(f0[t])))
            if PIANO_LOW <= midi_pitch < PIANO_LOW + 88:
                key_idx = midi_pitch - PIANO_LOW
                probs[key_idx, t] = float(voiced_prob[t])
                current_pitch_idx = key_idx
        elif current_pitch_idx is not None and t not in onset_set:
            # Sustain the last detected pitch until the next onset
            # (handles the decay phase where pYIN loses voicing)
            probs[current_pitch_idx, t] = 0.3   # low confidence for sustained frames
        else:
            current_pitch_idx = None   # silence or new onset resets the note

    return (probs >= threshold).astype(np.float32), probs


# ---------------------------------------------------------------------------
# Ground truth: MIDI → piano roll
# ---------------------------------------------------------------------------

def midi_to_piano_roll(midi_path: str, n_frames: int) -> np.ndarray:
    """
    Convert a MIDI file to an 88-key binary piano roll aligned to mel frames.

    Parameters
    ----------
    midi_path : str
        Path to the .midi / .mid file.
    n_frames : int
        Number of time frames to produce (must match the predicted piano roll).

    Returns
    -------
    piano_roll : (88, n_frames) float32
    """
    if not _HAS_MIDI:
        raise RuntimeError("pretty_midi is required to load ground-truth MIDI. "
                           "Install with: pip install pretty_midi")

    pm          = pretty_midi.PrettyMIDI(midi_path)
    frame_dur   = HOP_LENGTH / SR
    piano_roll  = np.zeros((88, n_frames), dtype=np.float32)

    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            pitch = note.pitch
            if not (PIANO_LOW <= pitch < PIANO_LOW + 88):
                continue
            key_idx   = pitch - PIANO_LOW
            start_fr  = int(note.start / frame_dur)
            end_fr    = min(int(note.end   / frame_dur) + 1, n_frames)
            piano_roll[key_idx, start_fr:end_fr] = 1.0

    return piano_roll


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def _draw_roll(ax, piano_roll: np.ndarray, title: str, dur_sec: float,
               cmap: str = 'Blues') -> None:
    """Draw a single piano roll onto an existing Axes."""
    ax.imshow(piano_roll, aspect='auto', origin='lower',
              extent=[0, dur_sec, PIANO_LOW, PIANO_LOW + 88],
              cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    ax.set_ylabel('MIDI pitch')
    ax.set_title(title, fontsize=11, fontweight='bold')
    for pitch in range(24, 109, 12):          # octave lines
        ax.axhline(pitch, color='gray', linewidth=0.4, alpha=0.5)
    # Label C notes (C2=36, C3=48, …)
    note_names = {36: 'C2', 48: 'C3', 60: 'C4 (middle)',
                  72: 'C5', 84: 'C6', 96: 'C7'}
    for pitch, name in note_names.items():
        if PIANO_LOW <= pitch < PIANO_LOW + 88:
            ax.text(-0.3, pitch, name, va='center', ha='right',
                    fontsize=6, color='dimgray',
                    transform=ax.get_yaxis_transform())


def save_piano_roll_png(piano_roll: np.ndarray, path: str) -> None:
    """Save a single predicted piano roll."""
    T       = piano_roll.shape[1]
    dur_sec = T * HOP_LENGTH / SR
    fig, ax = plt.subplots(figsize=(max(12, dur_sec * 0.5), 5))
    _draw_roll(ax, piano_roll, 'Predicted Piano Roll', dur_sec)
    ax.set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Piano roll PNG saved → {path}")


def save_comparison_png(predicted: np.ndarray, ground_truth: np.ndarray,
                        path: str, model_type: str) -> None:
    """
    Save a stacked comparison image:
        top    — Ground Truth (from MIDI)
        bottom — Predicted    (from model)
    Includes a per-frame F1 score in the title.
    """
    T       = predicted.shape[1]
    dur_sec = T * HOP_LENGTH / SR

    # Compute frame-level metrics for the title
    tp = (predicted * ground_truth).sum()
    fp = (predicted * (1 - ground_truth)).sum()
    fn = ((1 - predicted) * ground_truth).sum()
    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    f1   = 2 * prec * rec / (prec + rec + 1e-6)

    model_label = {
        'traditional':    'Traditional SP  (pYIN + onset detection)',
        'cnn_bilstm':     'CNN + BiLSTM',
        'cnn_transformer':'CNN + Transformer',
    }.get(model_type, model_type)

    fig_w = max(14, dur_sec * 0.5)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 9),
                             sharex=True, sharey=True)
    fig.suptitle(
        f'Piano Roll Comparison  —  {model_label}\n'
        f'Frame-level  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}',
        fontsize=12, fontweight='bold', y=1.01
    )

    _draw_roll(axes[0], ground_truth, 'Ground Truth  (MIDI)', dur_sec, cmap='Greens')
    _draw_roll(axes[1], predicted,    'Predicted     (Model)', dur_sec, cmap='Blues')
    axes[1].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison PNG saved → {path}")
    print(f"  Frame-level  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")


def save_midi(piano_roll: np.ndarray, path: str) -> None:
    if not _HAS_MIDI:
        print("pretty_midi not installed — skipping MIDI export. "
              "Install with: pip install pretty_midi")
        return

    pm    = pretty_midi.PrettyMIDI()
    instr = pretty_midi.Instrument(program=0, name='Piano')
    frame_dur = HOP_LENGTH / SR

    for key_idx in range(88):
        midi_pitch = key_idx + PIANO_LOW
        active     = piano_roll[key_idx]
        in_note    = False
        note_start = 0

        for t, val in enumerate(active):
            if val and not in_note:
                in_note    = True
                note_start = t
            elif not val and in_note:
                in_note = False
                s = note_start * frame_dur
                e = t * frame_dur
                if e > s:
                    instr.notes.append(
                        pretty_midi.Note(velocity=80, pitch=midi_pitch,
                                         start=s, end=e))
        if in_note:
            s = note_start * frame_dur
            e = len(active) * frame_dur
            instr.notes.append(
                pretty_midi.Note(velocity=80, pitch=midi_pitch, start=s, end=e))

    pm.instruments.append(instr)
    pm.write(path)
    print(f"MIDI saved → {path}")


def print_stats(piano_roll: np.ndarray) -> None:
    density  = piano_roll.mean()
    polyphony = piano_roll.sum(axis=0)          # notes active per frame
    active_frames = (polyphony > 0).mean()
    print(f"  Note density   : {density*100:.2f}%")
    print(f"  Active frames  : {active_frames*100:.1f}%")
    print(f"  Avg polyphony  : {polyphony[polyphony > 0].mean():.2f} notes/frame"
          if active_frames > 0 else "  Avg polyphony  : 0")
    # Most-predicted pitches
    pitch_totals = piano_roll.sum(axis=1)
    top5 = np.argsort(pitch_totals)[::-1][:5]
    note_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    print("  Top 5 pitches  :", ', '.join(
        f"{note_names[(p + 9) % 12]}{(p + 21) // 12 - 1}({p+21})"
        for p in top5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Transcribe an audio file to a piano roll using a trained model '
                    'or traditional signal processing.')
    parser.add_argument('--audio', required=True,
                        help='Path to input audio file (.wav, .mp3, .flac, …)')
    parser.add_argument('--model-path', default=None,
                        help='Path to saved model checkpoint (.pt). '
                             'Required for cnn_bilstm and cnn_transformer. '
                             'Not needed for --model-type traditional.')
    parser.add_argument('--model-type', default='cnn_transformer',
                        choices=['traditional', 'cnn_bilstm', 'cnn_transformer'],
                        help='Transcription method to use.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for note activation (default 0.5). '
                             'Lower values detect more notes but increase false positives.')
    parser.add_argument('--output-dir', default='outputs/transcriptions',
                        help='Directory for output files')
    parser.add_argument('--midi-path', default=None,
                        help='Path to ground-truth MIDI file. When provided, saves a '
                             'stacked comparison image (ground truth on top, '
                             'predicted on bottom) instead of a single piano roll.')
    parser.add_argument('--save-midi', action='store_true',
                        help='Also export a .mid file')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Validate: learned models need a checkpoint
    if args.model_type != 'traditional' and args.model_path is None:
        sys.exit(f"--model-path is required for --model-type {args.model_type}")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ---- Load & process audio ----
    if not os.path.isfile(args.audio):
        sys.exit(f"Audio file not found: {args.audio}")

    print(f"\nLoading audio: {args.audio}")
    mel = audio_to_mel(args.audio)
    n_frames = mel.shape[-1]
    duration = n_frames * HOP_LENGTH / SR
    print(f"Duration : {duration:.1f} s  |  Mel shape : {mel.shape}")

    # ---- Transcribe ----
    print(f"\nTranscribing with: {args.model_type}  (threshold={args.threshold}) …")

    if args.model_type == 'traditional':
        piano_roll_bin, piano_roll_probs = transcribe_traditional(
            args.audio, n_frames, threshold=args.threshold)
    else:
        mel_norm = normalize(mel)
        model = get_model(args.model_type).to(device)
        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print(f"  Checkpoint : {args.model_path}")
        print(f"  Device     : {device}")
        piano_roll_bin, piano_roll_probs = transcribe(
            model, mel_norm, device=device, threshold=args.threshold)

    print("\nTranscription statistics:")
    print_stats(piano_roll_bin)

    # ---- Save outputs ----
    stem = os.path.splitext(os.path.basename(args.audio))[0]
    tag  = f"{stem}_{args.model_type}"

    if args.midi_path:
        # Load ground truth and save stacked comparison image
        if not os.path.isfile(args.midi_path):
            sys.exit(f"MIDI file not found: {args.midi_path}")
        print(f"\nLoading ground-truth MIDI: {args.midi_path}")
        ground_truth = midi_to_piano_roll(args.midi_path, piano_roll_bin.shape[1])
        print("\nGround truth statistics:")
        print_stats(ground_truth)
        cmp_path = os.path.join(args.output_dir, f"{tag}_comparison.png")
        save_comparison_png(piano_roll_bin, ground_truth, cmp_path, args.model_type)
    else:
        png_path = os.path.join(args.output_dir, f"{tag}_piano_roll.png")
        save_piano_roll_png(piano_roll_bin, png_path)

    if args.save_midi:
        out_midi = os.path.join(args.output_dir, f"{tag}.mid")
        save_midi(piano_roll_bin, out_midi)

    print("\nDone.")


if __name__ == '__main__':
    main()
