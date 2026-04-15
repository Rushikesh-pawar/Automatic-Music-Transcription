import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

from dataset import get_dataloaders
from model import get_model, count_parameters

def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')

        torch.backends.cudnn.benchmark = True
        print(f"Using GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("CUDA not available — using CPU")
    return device

def compute_metrics(probs, target, threshold=0.5):

    if target.dim() == 4:
        target = target.squeeze(1)
    pred = (probs > threshold).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn}

def train_epoch(model, train_loader, optimizer, criterion, device, scaler):

    model.train()
    total_loss = 0.0
    use_amp = device.type == 'cuda'

    pbar = tqdm(train_loader, desc='Train')
    for batch in pbar:
        mel       = batch['mel'].to(device, non_blocking=True)
        pianoroll = batch['pianoroll'].to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits, probs = model(mel)

            if pianoroll.shape[-1] != logits.shape[-1]:
                pianoroll = torch.nn.functional.interpolate(
                    pianoroll, size=logits.shape[-1], mode='nearest')

            target = pianoroll.squeeze(1)
            loss   = criterion(logits, target)

        optimizer.zero_grad()

        if use_amp:

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(train_loader)

@torch.no_grad()
def eval_epoch(model, val_loader, criterion, device):

    model.eval()
    total_loss  = 0.0
    all_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    pbar = tqdm(val_loader, desc='Val  ')
    for batch in pbar:
        mel       = batch['mel'].to(device, non_blocking=True)
        pianoroll = batch['pianoroll'].to(device, non_blocking=True)

        logits, probs = model(mel)

        if pianoroll.shape[-1] != logits.shape[-1]:
            pianoroll = torch.nn.functional.interpolate(
                pianoroll, size=logits.shape[-1], mode='nearest')

        target = pianoroll.squeeze(1)
        loss   = criterion(logits, target)
        total_loss += loss.item()

        metrics = compute_metrics(probs, target)
        for k in all_metrics:
            all_metrics[k] += metrics[k]

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    n = len(val_loader)
    return total_loss / n, {k: v / n for k, v in all_metrics.items()}

def main():
    parser = argparse.ArgumentParser(description='Train music transcription model')
    parser.add_argument('--pairs-csv',      type=str, required=True)
    parser.add_argument('--mels-dir',       type=str, required=True)
    parser.add_argument('--pianorolls-dir', type=str, required=True)
    parser.add_argument('--epochs',         type=int,   default=20)
    parser.add_argument('--batch-size',     type=int,   default=16)
    parser.add_argument('--lr',             type=float, default=0.001)
    parser.add_argument('--num-workers',    type=int,   default=4,
                        help='DataLoader worker processes (0 = main process)')
    parser.add_argument('--output-dir',     type=str,   default='outputs/checkpoints')
    parser.add_argument('--train-split',    type=float, default=0.8)
    parser.add_argument('--pos-weight',     type=float, default=19.0,
                        help='BCEWithLogitsLoss pos_weight (~(1-density)/density)')
    parser.add_argument('--resume',         type=str,   default=None)
    parser.add_argument('--model-type',     type=str,   default='cnn_bilstm',
                        choices=['traditional', 'cnn_bilstm', 'cnn_transformer'])
    parser.add_argument('--no-amp',         action='store_true',
                        help='Disable Automatic Mixed Precision (AMP)')
    args = parser.parse_args()

    device = setup_device()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading data from {args.pairs_csv} ...")
    train_loader, val_loader, _ = get_dataloaders(
        args.pairs_csv, args.mels_dir, args.pianorolls_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=args.train_split,
    )

    model = get_model(args.model_type).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")
    print(f"Model      : {args.model_type}  |  Parameters: {count_parameters(model):,}")

    optimizer  = optim.Adam(model.parameters(), lr=args.lr)
    pos_weight = torch.tensor(args.pos_weight).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    use_amp = device.type == 'cuda' and not args.no_amp
    scaler  = GradScaler(device='cuda', enabled=use_amp)
    print(f"AMP        : {'enabled' if use_amp else 'disabled'}")

    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    best_val_f1 = 0.0
    best_epoch  = 0

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        train_loss             = train_epoch(model, train_loader, optimizer,
                                             criterion, device, scaler)
        val_loss, val_metrics  = eval_epoch(model, val_loader, criterion, device)

        print(f"Train loss : {train_loss:.4f}")
        print(f"Val loss   : {val_loss:.4f}")
        print(f"Val metrics: P={val_metrics['precision']:.4f}  "
              f"R={val_metrics['recall']:.4f}  F1={val_metrics['f1']:.4f}")

        writer.add_scalars('loss',    {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('metrics', val_metrics, epoch)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('lr', current_lr, epoch)

        scheduler.step(val_loss)

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch  = epoch
            ckpt = os.path.join(args.output_dir, f'best_{args.model_type}.pt')
            torch.save(model.state_dict(), ckpt)
            print(f"Saved best model → {ckpt}  (F1={best_val_f1:.4f})")

        latest = os.path.join(args.output_dir, f'latest_{args.model_type}.pt')
        torch.save(model.state_dict(), latest)

    writer.close()
    print(f"\nBest validation F1 : {best_val_f1:.4f} (epoch {best_epoch + 1})")
    print(f"Checkpoints saved  → {args.output_dir}")

if __name__ == '__main__':
    main()