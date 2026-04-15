import argparse
import os
import sys

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from model import get_model

try:
    import pretty_midi
    _HAS_MIDI = True
except ImportError:
    _HAS_MIDI = False

SR         = 22050
N_MELS     = 128
N_FFT      = 2048
HOP_LENGTH = 512
FMAX       = 8000
CHUNK_SIZE = 256
PIANO_LOW  = 21

def build_mel_pipeline(device):

    mel_transform = T.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, f_max=FMAX,
    ).to(device)
    amp_to_db = T.AmplitudeToDB(stype='power', top_db=80).to(device)
    return mel_transform, amp_to_db

def audio_to_mel(audio_path: str, device: torch.device) -> np.ndarray:

    mel_transform, amp_to_db = build_mel_pipeline(device)

    waveform, orig_sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0, keepdim=True)

    if orig_sr != SR:
        waveform = torchaudio.functional.resample(waveform, orig_sr, SR)

    waveform = waveform.to(device)

    with torch.no_grad():
        mel   = mel_transform(waveform)
        mel_db = amp_to_db(mel)

    return mel_db.squeeze(0).cpu().numpy()

def normalize(mel: np.ndarray) -> np.ndarray:
    lo, hi = mel.min(), mel.max()
    return (mel - lo) / (hi - lo + 1e-8)

def transcribe(model, mel: np.ndarray, device: torch.device,
               threshold: float = 0.5):

    model.eval()
    T_len = mel.shape[-1]
    probs_full = np.zeros((88, T_len), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, T_len, CHUNK_SIZE):
            end       = min(start + CHUNK_SIZE, T_len)
            chunk     = mel[:, start:end].copy()
            chunk_len = chunk.shape[-1]

            if chunk_len < CHUNK_SIZE:
                chunk = np.pad(chunk, ((0, 0), (0, CHUNK_SIZE - chunk_len)))

            x = (torch.from_numpy(chunk)
                 .unsqueeze(0).unsqueeze(0)
                 .to(device, non_blocking=True))

            _, p = model(x)
            probs_full[:, start:end] = p[0].cpu().numpy()[:, :chunk_len]

    return (probs_full >= threshold).astype(np.float32), probs_full

def transcribe_traditional(audio_path: str, n_frames: int,
                            threshold: float = 0.5):

    print("  Running pYIN pitch detection (CPU) …")
    y, _ = librosa.load(audio_path, sr=SR, mono=True)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=librosa.note_to_hz('A0'), fmax=librosa.note_to_hz('C8'),
        sr=SR, hop_length=HOP_LENGTH, frame_length=N_FFT,
    )
    onsets    = librosa.onset.onset_detect(y=y, sr=SR, hop_length=HOP_LENGTH,
                                           units='frames')
    onset_set = set(onsets.tolist())

    probs             = np.zeros((88, n_frames), dtype=np.float32)
    T_len             = min(len(f0), n_frames)
    current_pitch_idx = None

    for t in range(T_len):
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

    return (probs >= threshold).astype(np.float32), probs

def midi_to_piano_roll(midi_path: str, n_frames: int) -> np.ndarray:
    if not _HAS_MIDI:
        raise RuntimeError("pretty_midi required: pip install pretty_midi")
    pm         = pretty_midi.PrettyMIDI(midi_path)
    frame_dur  = HOP_LENGTH / SR
    piano_roll = np.zeros((88, n_frames), dtype=np.float32)
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            if not (PIANO_LOW <= note.pitch < PIANO_LOW + 88):
                continue
            key_idx  = note.pitch - PIANO_LOW
            start_fr = int(note.start / frame_dur)
            end_fr   = min(int(note.end / frame_dur) + 1, n_frames)
            piano_roll[key_idx, start_fr:end_fr] = 1.0
    return piano_roll

def _draw_roll(ax, piano_roll, title, dur_sec, cmap='Blues'):
    ax.imshow(piano_roll, aspect='auto', origin='lower',
              extent=[0, dur_sec, PIANO_LOW, PIANO_LOW + 88],
              cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    ax.set_ylabel('MIDI pitch')
    ax.set_title(title, fontsize=11, fontweight='bold')
    for pitch in range(24, 109, 12):
        ax.axhline(pitch, color='gray', linewidth=0.4, alpha=0.5)
    note_names = {36:'C2', 48:'C3', 60:'C4 (middle)', 72:'C5', 84:'C6', 96:'C7'}
    for pitch, name in note_names.items():
        if PIANO_LOW <= pitch < PIANO_LOW + 88:
            ax.text(-0.3, pitch, name, va='center', ha='right',
                    fontsize=6, color='dimgray',
                    transform=ax.get_yaxis_transform())

def save_piano_roll_png(piano_roll, path):
    T_len   = piano_roll.shape[1]
    dur_sec = T_len * HOP_LENGTH / SR
    fig, ax = plt.subplots(figsize=(max(12, dur_sec * 0.5), 5))
    _draw_roll(ax, piano_roll, 'Predicted Piano Roll', dur_sec)
    ax.set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Piano roll PNG saved → {path}")

def save_comparison_png(predicted, ground_truth, path, model_type):
    T_len   = predicted.shape[1]
    dur_sec = T_len * HOP_LENGTH / SR
    tp  = (predicted * ground_truth).sum()
    fp  = (predicted * (1 - ground_truth)).sum()
    fn  = ((1 - predicted) * ground_truth).sum()
    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    f1   = 2 * prec * rec / (prec + rec + 1e-6)
    label = {'traditional': 'Traditional SP (pYIN)',
             'cnn_bilstm': 'CNN + BiLSTM',
             'cnn_transformer': 'CNN + Transformer'}.get(model_type, model_type)
    fig, axes = plt.subplots(2, 1, figsize=(max(14, dur_sec * 0.5), 9),
                             sharex=True, sharey=True)
    fig.suptitle(f'Piano Roll Comparison — {label}\n'
                 f'Frame-level  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}',
                 fontsize=12, fontweight='bold', y=1.01)
    _draw_roll(axes[0], ground_truth, 'Ground Truth  (MIDI)',  dur_sec, cmap='Greens')
    _draw_roll(axes[1], predicted,    'Predicted     (Model)', dur_sec, cmap='Blues')
    axes[1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison PNG saved → {path}")
    print(f"  Frame-level  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

def save_midi(piano_roll, path):
    if not _HAS_MIDI:
        print("pretty_midi not installed — skipping MIDI export.")
        return
    pm        = pretty_midi.PrettyMIDI()
    instr     = pretty_midi.Instrument(program=0, name='Piano')
    frame_dur = HOP_LENGTH / SR
    for key_idx in range(88):
        midi_pitch = key_idx + PIANO_LOW
        active     = piano_roll[key_idx]
        in_note    = False
        note_start = 0
        for t, val in enumerate(active):
            if val and not in_note:
                in_note = True; note_start = t
            elif not val and in_note:
                in_note = False
                s, e = note_start * frame_dur, t * frame_dur
                if e > s:
                    instr.notes.append(pretty_midi.Note(80, midi_pitch, s, e))
        if in_note:
            s, e = note_start * frame_dur, len(active) * frame_dur
            instr.notes.append(pretty_midi.Note(80, midi_pitch, s, e))
    pm.instruments.append(instr)
    pm.write(path)
    print(f"MIDI saved → {path}")

def print_stats(piano_roll):
    density   = piano_roll.mean()
    polyphony = piano_roll.sum(axis=0)
    active    = (polyphony > 0).mean()
    print(f"  Note density   : {density*100:.2f}%")
    print(f"  Active frames  : {active*100:.1f}%")
    if active > 0:
        print(f"  Avg polyphony  : {polyphony[polyphony > 0].mean():.2f} notes/frame")
    note_names = ['A','As','B','C','Cs','D','Ds','E','F','Fs','G','Gs']
    top5 = np.argsort(piano_roll.sum(axis=1))[::-1][:5]
    print("  Top 5 pitches  :", ', '.join(
        f"{note_names[(p+9)%12]}{(p+21)//12-1}({p+21})" for p in top5))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio',       required=True)
    parser.add_argument('--model-path',  default=None)
    parser.add_argument('--model-type',  default='cnn_transformer',
                        choices=['traditional', 'cnn_bilstm', 'cnn_transformer'])
    parser.add_argument('--threshold',   type=float, default=0.5)
    parser.add_argument('--output-dir',  default='outputs/transcriptions')
    parser.add_argument('--midi-path',   default=None)
    parser.add_argument('--save-midi',   action='store_true')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if args.model_type != 'traditional' and args.model_path is None:
        sys.exit(f"--model-path is required for --model-type {args.model_type}")

    if not os.path.isfile(args.audio):
        sys.exit(f"Audio file not found: {args.audio}")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}")

    print(f"\nLoading audio: {args.audio}")
    mel      = audio_to_mel(args.audio, device)
    n_frames = mel.shape[-1]
    duration = n_frames * HOP_LENGTH / SR
    print(f"Duration : {duration:.1f} s  |  Mel shape : {mel.shape}")

    print(f"\nTranscribing with: {args.model_type}  (threshold={args.threshold}) …")

    if args.model_type == 'traditional':
        piano_roll_bin, _ = transcribe_traditional(args.audio, n_frames, args.threshold)
    else:
        mel_norm = normalize(mel)
        model    = get_model(args.model_type).to(device)
        state    = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print(f"  Checkpoint : {args.model_path}  |  Device: {device}")
        piano_roll_bin, _ = transcribe(model, mel_norm, device, args.threshold)

    print("\nTranscription statistics:")
    print_stats(piano_roll_bin)

    stem = os.path.splitext(os.path.basename(args.audio))[0]
    tag  = f"{stem}_{args.model_type}"

    if args.midi_path:
        if not os.path.isfile(args.midi_path):
            sys.exit(f"MIDI file not found: {args.midi_path}")
        print(f"\nLoading ground-truth MIDI: {args.midi_path}")
        ground_truth = midi_to_piano_roll(args.midi_path, piano_roll_bin.shape[1])
        print("\nGround truth statistics:")
        print_stats(ground_truth)
        save_comparison_png(piano_roll_bin, ground_truth,
                            os.path.join(args.output_dir, f"{tag}_comparison.png"),
                            args.model_type)
    else:
        save_piano_roll_png(piano_roll_bin,
                            os.path.join(args.output_dir, f"{tag}_piano_roll.png"))

    if args.save_midi:
        save_midi(piano_roll_bin, os.path.join(args.output_dir, f"{tag}.mid"))

    print("\nDone.")

if __name__ == '__main__':
    main()