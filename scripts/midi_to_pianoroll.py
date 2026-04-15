import argparse
from pathlib import Path
import numpy as np
import pretty_midi
from tqdm import tqdm


PITCH_LOW  = 21
PITCH_HIGH = 108


def midi_to_88_pr(midi_path, sr=22050, hop_length=512):
    pm    = pretty_midi.PrettyMIDI(str(midi_path))
    fs    = float(sr) / float(hop_length)
    pr128 = pm.get_piano_roll(fs=fs)
    pr88  = pr128[PITCH_LOW:PITCH_HIGH + 1, :]
    return (pr88 > 0).astype(np.uint8)


def align_time_frames(pr, target_T):
    T = pr.shape[1]
    if T == target_T:
        return pr
    if T > target_T:
        return pr[:, :target_T]
    pad = np.zeros((pr.shape[0], target_T - T), dtype=pr.dtype)
    return np.concatenate([pr, pad], axis=1)


def main():
    parser = argparse.ArgumentParser(description='Convert MIDI to 88-key piano-rolls aligned to mel frames')
    parser.add_argument('--midi-dir',    type=Path, default=Path('Audio Files/2015'))
    parser.add_argument('--mel-dir',     type=Path, default=Path('mels'))
    parser.add_argument('--out-dir',     type=Path, default=Path('pianorolls'))
    parser.add_argument('--sr',          type=int,  default=22050)
    parser.add_argument('--hop-length',  type=int,  default=512)
    parser.add_argument('--save-npz',    action='store_true')
    parser.add_argument('--exts',        nargs='+', default=['.midi', '.mid'])
    args = parser.parse_args()

    files = []
    for ext in args.exts:
        files.extend(list(args.midi_dir.rglob(f'*{ext}')))
    files = sorted(files)
    if not files:
        print('No MIDI files found')
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = []

    for m in tqdm(files, desc='MIDI->pianoroll'):
        try:
            pr88 = midi_to_88_pr(m, sr=args.sr, hop_length=args.hop_length)
            rel  = m.relative_to(args.midi_dir)
            out_base = args.out_dir.joinpath(rel).with_suffix('')
            out_base.parent.mkdir(parents=True, exist_ok=True)
            pr_path  = out_base.with_suffix('.npy')

            mel_path = args.mel_dir.joinpath(rel).with_suffix('.npy')
            if args.mel_dir.exists() and mel_path.exists():
                mel        = np.load(mel_path)
                pr_aligned = align_time_frames(pr88, mel.shape[1])
                np.save(pr_path, pr_aligned)
                if args.save_npz:
                    npz_path = out_base.with_suffix('.npz')
                    np.savez_compressed(npz_path,
                                        mel=mel[:, :pr_aligned.shape[1]],
                                        piano_roll=pr_aligned)
                pairs.append((str(m), str(pr_path), str(mel_path)))
            else:
                np.save(pr_path, pr88)
                pairs.append((str(m), str(pr_path), ''))
        except Exception as e:
            print(f'Error processing {m}: {e}')

    csv_path = args.out_dir.joinpath('pairs.csv')
    with open(csv_path, 'w') as fh:
        fh.write('midi,piano_roll,mel\n')
        for a, b, c in pairs:
            fh.write(f'"{a}","{b}","{c}"\n')

    print('Done. Piano-rolls saved to', args.out_dir)


if __name__ == '__main__':
    main()