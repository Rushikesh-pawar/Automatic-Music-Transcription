import argparse
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import pretty_midi


def compute_mel(path, sr=22050, n_fft=2048, hop_length=512, n_mels=128, fmax=8000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels, fmax=fmax
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db, y, sr


def save_mel(mel, out_path, save_png=False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path.with_suffix(out_path.suffix + '.npy'), mel)
    if save_png:
        plt.figure(figsize=(6, 4))
        plt.imshow(mel, origin='lower', aspect='auto', cmap='magma')
        plt.axis('off')
        plt.tight_layout(pad=0)
        png_path = out_path.with_suffix(out_path.suffix + '.png')
        plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
        plt.close()


def audio_to_midi(y, sr, midi_out_path):
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )

    midi_pitch  = librosa.hz_to_midi(f0)
    onsets      = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onsets, sr=sr)

    midi       = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    for i in range(len(onsets) - 1):
        start_frame = onsets[i]
        end_frame   = onsets[i + 1]
        start_time  = onset_times[i]
        end_time    = onset_times[i + 1]

        segment = midi_pitch[start_frame:end_frame]
        segment = segment[~np.isnan(segment)]

        if len(segment) == 0:
            continue

        pitch = int(np.median(segment))
        note  = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=start_time,
            end=end_time
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi_out_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(midi_out_path))


def main():
    parser = argparse.ArgumentParser(description='Convert audio to mel spectrograms + MIDI.')
    parser.add_argument('--input-dir',  type=Path, default=Path('Audio Files/2015'))
    parser.add_argument('--mel-dir',    type=Path, default=Path('mels'))
    parser.add_argument('--midi-dir',   type=Path, default=Path('predicted_midis'))
    parser.add_argument('--sr',         type=int,  default=22050)
    parser.add_argument('--n-mels',     type=int,  default=128)
    parser.add_argument('--n-fft',      type=int,  default=2048)
    parser.add_argument('--hop-length', type=int,  default=512)
    parser.add_argument('--fmax',       type=int,  default=8000)
    parser.add_argument('--save-png',   action='store_true')
    parser.add_argument('--exts',       nargs='+', default=['.wav', '.mp3', '.flac'])
    args = parser.parse_args()

    files = []
    for ext in args.exts:
        files.extend(list(args.input_dir.rglob(f'*{ext}')))
    files = sorted(files)

    if not files:
        print(f'No audio files found in {args.input_dir}')
        return

    args.mel_dir.mkdir(parents=True, exist_ok=True)
    args.midi_dir.mkdir(parents=True, exist_ok=True)

    for f in tqdm(files, desc='Processing'):
        try:
            mel, y, sr = compute_mel(
                f,
                sr=args.sr,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                n_mels=args.n_mels,
                fmax=args.fmax
            )
            rel      = f.relative_to(args.input_dir)
            mel_out  = args.mel_dir.joinpath(rel).with_suffix('')
            save_mel(mel, mel_out, save_png=args.save_png)

            midi_out = args.midi_dir.joinpath(rel).with_suffix('.mid')
            audio_to_midi(y, sr, midi_out)

        except Exception as e:
            print(f'Error processing {f}: {e}')


if __name__ == '__main__':
    main()