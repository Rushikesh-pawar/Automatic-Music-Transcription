import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class MusicTranscriptionDataset(Dataset):

    def __init__(self, pairs_csv_or_df, mels_dir, pianorolls_dir,
                 normalize=True, chunk_size=256):
        self.mels_dir = mels_dir
        self.pianorolls_dir = pianorolls_dir
        self.normalize = normalize
        self.chunk_size = chunk_size

        pairs_df = (pd.read_csv(pairs_csv_or_df)
                    if isinstance(pairs_csv_or_df, str)
                    else pairs_csv_or_df)

        self.chunks = []
        for _, row in pairs_df.iterrows():
            mel_file = row['mel']
            pr_file  = row['piano_roll']
            mel_path = (mel_file if os.path.isabs(mel_file)
                        else os.path.join(mels_dir, mel_file))
            pr_path  = (pr_file if os.path.isabs(pr_file)
                        else os.path.join(pianorolls_dir, pr_file))

            mel = np.load(mel_path, mmap_mode='r')
            pr  = np.load(pr_path,  mmap_mode='r')
            t   = min(mel.shape[-1], pr.shape[-1])

            if t < chunk_size:
                self.chunks.append((mel_path, pr_path, 0, t))
            else:
                for start in range(0, t - chunk_size + 1, chunk_size):
                    self.chunks.append((mel_path, pr_path, start, chunk_size))

        print(f"Indexed {len(self.chunks)} chunks from {len(pairs_df)} files")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        mel_path, pr_path, start, length = self.chunks[idx]

        mel = np.load(mel_path).astype(np.float32)
        pr  = np.load(pr_path).astype(np.float32)

        mel = mel[..., start:start + length]
        pr  = pr[..., start:start + length]

        if length < self.chunk_size:
            pad = self.chunk_size - length
            mel = np.pad(mel, ((0, 0), (0, pad)), mode='constant')
            pr  = np.pad(pr,  ((0, 0), (0, pad)), mode='constant')

        if self.normalize and mel.size > 0:
            mel_min, mel_max = mel.min(), mel.max()
            if mel_max > mel_min:
                mel = (mel - mel_min) / (mel_max - mel_min)

        mel = np.expand_dims(mel, axis=0)
        pr  = np.expand_dims(pr,  axis=0)

        return {
            'mel':            torch.from_numpy(mel),
            'pianoroll':      torch.from_numpy(pr),
            'mel_path':       mel_path,
            'pianoroll_path': pr_path,
        }

def get_dataloaders(pairs_csv, mels_dir, pianorolls_dir, batch_size=16,
                    num_workers=0, train_split=0.8, normalize=True, seed=42,
                    chunk_size=256):

    np.random.seed(seed)
    torch.manual_seed(seed)

    all_files = pd.read_csv(pairs_csv)
    all_files = all_files.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_files   = len(all_files)
    n_train   = int(train_split * n_files)

    train_df = all_files.iloc[:n_train]
    val_df   = all_files.iloc[n_train:]

    train_dataset = MusicTranscriptionDataset(
        train_df, mels_dir, pianorolls_dir, normalize=normalize, chunk_size=chunk_size)
    val_dataset = MusicTranscriptionDataset(
        val_df, mels_dir, pianorolls_dir, normalize=normalize, chunk_size=chunk_size)

    print(f"File split : {n_train} train files, {n_files - n_train} val files")
    print(f"Chunk split: {len(train_dataset)} train chunks, {len(val_dataset)} val chunks")

    cuda_available = torch.cuda.is_available()

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=cuda_available,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)

    return train_loader, val_loader, train_dataset