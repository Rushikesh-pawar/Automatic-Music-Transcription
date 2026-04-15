import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_pairs(pairs_csv_path):
    pairs = []
    with open(pairs_csv_path, 'r') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            pairs.append((r['midi'], r['piano_roll'], r['mel']))
    return pairs


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def visualize_pair(mel_path, pr_path, out_path):
    mel = np.load(mel_path)
    pr  = np.load(pr_path)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6),
                                   gridspec_kw={'height_ratios': [3, 1]})
    ax1.imshow(mel, origin='lower', aspect='auto', cmap='magma')
    ax1.set_title('Mel spectrogram')
    ax1.axis('off')
    ax2.imshow(pr, origin='lower', aspect='auto', cmap='gray_r')
    ax2.set_title('88-key piano-roll (binary)')
    ax2.set_ylabel('MIDI 21->108')
    ax2.set_xlabel('Time frames')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(pairs_csv='pianorolls/pairs.csv', out_dir='outputs', num_samples=6):
    pairs   = load_pairs(pairs_csv)
    out_dir = Path(out_dir)
    vis_dir = out_dir / 'visualizations'
    ensure_dir(vis_dir)

    stats        = defaultdict(list)
    pitch_sums   = np.zeros(88, dtype=np.float64)
    total_active = 0
    total_frames = 0
    sampled      = 0

    for midi_path, pr_path, mel_path in pairs:
        pr_p  = Path(pr_path)
        mel_p = Path(mel_path)
        if not pr_p.exists():
            continue

        pr    = np.load(pr_p).astype(np.float64)
        Tpr   = pr.shape[1]
        stats['pr_frames'].append(Tpr)
        total_frames += Tpr
        total_active += int(pr.sum())
        pitch_sums   += pr.sum(axis=1)
        stats['polyphony_mean'].append(pr.sum(axis=0).mean())

        if mel_p.exists() and sampled < num_samples:
            mel = np.load(mel_p)
            stats['mel_frames'].append(mel.shape[1])
            out_png = vis_dir / (pr_p.stem + '_pair.png')
            visualize_pair(mel_p, pr_p, out_png)
            sampled += 1

    n_files  = len(stats['pr_frames'])
    density  = total_active / (88 * total_frames) if total_frames > 0 else 0
    poly_mean = float(np.mean(stats['polyphony_mean'])) if stats['polyphony_mean'] else 0.0

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(21, 21 + 88), pitch_sums)
    ax.set_xlabel('MIDI pitch')
    ax.set_ylabel('Total active frames')
    ax.set_title('Pitch activation histogram (21..108)')
    plt.tight_layout()
    pitch_hist_path = out_dir / 'pitch_hist.png'
    fig.savefig(pitch_hist_path, dpi=150)
    plt.close(fig)

    md = out_dir / 'dataset_analysis.md'
    with open(md, 'w') as fh:
        fh.write('# Dataset analysis\n\n')
        fh.write(f'- Number of MIDI files processed: {n_files}\n')
        fh.write(f'- Total frames (sum of piano-roll frames): {total_frames}\n')
        fh.write(f'- Average piano-roll frames per file: {np.mean(stats["pr_frames"]):.1f}\n')
        fh.write(f'- Mel frames sample mean: {np.mean(stats.get("mel_frames", [0])):.1f}\n')
        fh.write(f'- Average polyphony (notes per frame): {poly_mean:.3f}\n')
        fh.write(f'- Note density (fraction of active notes): {density:.6f}\n')
        fh.write('\n')
        fh.write('See visualizations in `outputs/visualizations/` and pitch histogram: `outputs/pitch_hist.png`.\n')

    print('Saved visualizations to', vis_dir)
    print('Saved analysis to', md)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs-csv',   type=str, default='pianorolls/pairs.csv')
    parser.add_argument('--out-dir',     type=str, default='outputs')
    parser.add_argument('--num-samples', type=int, default=6)
    args = parser.parse_args()
    run(pairs_csv=args.pairs_csv, out_dir=args.out_dir, num_samples=args.num_samples)