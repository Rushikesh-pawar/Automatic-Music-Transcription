import argparse
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_mel(path, sr=22050, n_fft=2048, hop_length=512, n_mels=128, fmax=8000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


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


def main():
    parser = argparse.ArgumentParser(description='Convert audio files to mel spectrograms (log-power).')
    parser.add_argument('--input-dir', type=Path, default=Path('Audio Files/2015'), help='Input folder containing audio files (default: Audio Files/2015)')
    parser.add_argument('--output-dir', type=Path, default=Path('mels'), help='Output folder for mel arrays')
    parser.add_argument('--sr', type=int, default=22050)
    parser.add_argument('--n-mels', type=int, default=128)
    parser.add_argument('--n-fft', type=int, default=2048)
    parser.add_argument('--hop-length', type=int, default=512)
    parser.add_argument('--fmax', type=int, default=8000)
    parser.add_argument('--save-png', action='store_true', help='Also save a PNG image of the mel')
    parser.add_argument('--exts', nargs='+', default=['.wav', '.mp3', '.flac'], help='Audio file extensions to search')
    args = parser.parse_args()

    files = []
    for ext in args.exts:
        files.extend(list(args.input_dir.rglob(f'*{ext}')))
    files = sorted(files)

    if not files:
        print(f'No audio files found in {args.input_dir} with extensions {args.exts}')
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for f in tqdm(files, desc='Processing'):
        try:
            mel = compute_mel(f, sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length, n_mels=args.n_mels, fmax=args.fmax)
            rel = f.relative_to(args.input_dir)
            out_path = args.output_dir.joinpath(rel).with_suffix('')
            save_mel(mel, out_path, save_png=args.save_png)
        except Exception as e:
            print(f'Error processing {f}: {e}')


if __name__ == '__main__':
    main()
