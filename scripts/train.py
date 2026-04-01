"""
Training script for music transcription model.
Usage:
    python scripts/train.py --pairs-csv pianorolls/pairs.csv \
                            --mels-dir mels \
                            --pianorolls-dir pianorolls \
                            --epochs 20 --batch-size 16 --lr 0.001
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from dataset import get_dataloaders
from model import get_model, count_parameters


def compute_metrics(probs, target, threshold=0.5):
    """
    Compute frame-level metrics (TP, FP, FN, precision, recall, F1).
    
    Args:
        probs: (batch, 88, time) - predicted probabilities [0, 1]
        target: (batch, 1, 88, time) or (batch, 88, time) - ground truth binary
        threshold: Classification threshold
        
    Returns:
        metrics dict with precision, recall, f1
    """
    # Ensure target has correct shape
    if target.dim() == 4:  # (batch, 1, 88, time)
        target = target.squeeze(1)
    
    # Ensure probs and target match spatial dims
    if probs.shape != target.shape:
        # Upsample/downsample as needed
        if probs.shape[2] != target.shape[1]:
            raise ValueError(f"Time mismatch: probs {probs.shape} vs target {target.shape}")
    
    pred = (probs > threshold).float()
    
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc='Train')
    for batch in pbar:
        mel = batch['mel'].to(device)  # (batch, 1, mel_bins, time)
        pianoroll = batch['pianoroll'].to(device)  # (batch, 1, 88, time)
        
        # Forward
        logits, probs = model(mel)
        
        # Resize target to match output time dimension if needed
        if pianoroll.shape[-1] != logits.shape[-1]:
            # Interpolate pianoroll to match model output
            pianoroll = torch.nn.functional.interpolate(
                pianoroll, size=logits.shape[-1], mode='nearest'
            )
        
        # Remove channel dim from target for loss
        target = pianoroll.squeeze(1)  # (batch, 88, time)
        
        # Loss
        loss = criterion(logits, target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)


@torch.no_grad()
def eval_epoch(model, val_loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    all_metrics = {'precision': 0, 'recall': 0, 'f1': 0}
    
    pbar = tqdm(val_loader, desc='Val')
    for batch in pbar:
        mel = batch['mel'].to(device)
        pianoroll = batch['pianoroll'].to(device)
        
        # Forward
        logits, probs = model(mel)
        
        # Resize target
        if pianoroll.shape[-1] != logits.shape[-1]:
            pianoroll = torch.nn.functional.interpolate(
                pianoroll, size=logits.shape[-1], mode='nearest'
            )
        
        target = pianoroll.squeeze(1)
        
        # Loss
        loss = criterion(logits, target)
        total_loss += loss.item()
        
        # Metrics
        metrics = compute_metrics(probs, target)
        for key in all_metrics:
            all_metrics[key] += metrics[key]
        
        pbar.set_postfix({'loss': loss.item()})
    
    n = len(val_loader)
    avg_loss = total_loss / n
    for key in all_metrics:
        all_metrics[key] /= n
    
    return avg_loss, all_metrics


def main():
    parser = argparse.ArgumentParser(description='Train music transcription model')
    parser.add_argument('--pairs-csv', type=str, required=True, 
                       help='Path to pairs.csv from midi_to_pianoroll.py')
    parser.add_argument('--mels-dir', type=str, required=True, 
                       help='Directory with mel .npy files')
    parser.add_argument('--pianorolls-dir', type=str, required=True, 
                       help='Directory with piano-roll .npy files')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device: cuda or cpu')
    parser.add_argument('--output-dir', type=str, default='outputs/checkpoints',
                       help='Directory for saving checkpoints and logs')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Fraction of data for training')
    parser.add_argument('--pos-weight', type=float, default=19.0,
                       help='Positive class weight for BCEWithLogitsLoss (combats class imbalance; ~(1-density)/density)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--model-type', type=str, default='cnn_bilstm',
                       choices=['traditional', 'cnn_bilstm', 'cnn_transformer'],
                       help='Model architecture to train')

    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Loading data from {args.pairs_csv}...")
    
    # Data
    train_loader, val_loader, dataset = get_dataloaders(
        args.pairs_csv, args.mels_dir, args.pianorolls_dir,
        batch_size=args.batch_size, train_split=args.train_split
    )
    
    # Model
    model = get_model(args.model_type).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")
    print(f"Model: {args.model_type}  |  Parameters: {count_parameters(model):,}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # pos_weight addresses class imbalance: piano-roll is ~5% active, so weight notes ~19x
    pos_weight = torch.tensor(args.pos_weight).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Tensorboard
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    # Training loop
    best_val_f1 = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = eval_epoch(model, val_loader, criterion, device)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss: {val_loss:.4f}")
        print(f"Val metrics: P={val_metrics['precision']:.4f}, R={val_metrics['recall']:.4f}, F1={val_metrics['f1']:.4f}")

        # Logging
        writer.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('metrics', val_metrics, epoch)

        # LR scheduling
        scheduler.step(val_loss)

        # Checkpoint on best F1 (the real metric for transcription)
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            checkpoint_path = os.path.join(args.output_dir,
                                           f'best_{args.model_type}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path} (F1={best_val_f1:.4f})")

        # Save latest checkpoint
        latest_path = os.path.join(args.output_dir, f'latest_{args.model_type}.pt')
        torch.save(model.state_dict(), latest_path)
    
    writer.close()
    print(f"\nBest validation F1: {best_val_f1:.4f} (epoch {best_epoch + 1})")
    print(f"Checkpoints saved to {args.output_dir}")


if __name__ == '__main__':
    main()
