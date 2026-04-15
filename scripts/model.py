import math
import numpy as np
import torch
import torch.nn as nn

class TraditionalSP(nn.Module):

    @staticmethod
    def _midi_to_freq(pitch: int) -> float:
        return 440.0 * (2.0 ** ((pitch - 69) / 12.0))

    def __init__(self, mel_bins: int = 128, num_keys: int = 88,
                 sr: int = 22050, fmax: float = 8000.0,
                 hps_harmonics: int = 3, threshold_percentile: float = 85.0):
        super().__init__()
        self.mel_bins = mel_bins
        self.num_keys = num_keys
        self.hps_harmonics = hps_harmonics
        self.threshold_percentile = threshold_percentile

        import librosa
        mel_freqs = librosa.mel_frequencies(n_mels=mel_bins, fmax=fmax)
        key_bins = []
        for k in range(num_keys):
            freq = self._midi_to_freq(k + 21)
            bin_idx = int(np.argmin(np.abs(mel_freqs - freq)))
            key_bins.append(min(bin_idx, mel_bins - 1))
        self.register_buffer('key_bins', torch.tensor(key_bins, dtype=torch.long))

    def forward(self, mel: torch.Tensor):

        if mel.dim() == 4:
            mel = mel.squeeze(1)

        spec = torch.exp(mel)

        hps = spec.clone()
        for h in range(2, self.hps_harmonics + 1):

            downsampled = spec[:, ::h, :]
            length = downsampled.shape[1]
            hps[:, :length, :] = hps[:, :length, :] * downsampled

        key_energies = hps[:, self.key_bins, :]

        q = self.threshold_percentile / 100.0

        B, K, Tlen = key_energies.shape
        flat = key_energies.permute(0, 2, 1).reshape(B * Tlen, K)
        thresh = torch.quantile(flat, q, dim=1, keepdim=True)
        probs_flat = (flat >= thresh).float()
        probs = probs_flat.reshape(B, Tlen, K).permute(0, 2, 1)

        return probs, probs

class CNNBiLSTM(nn.Module):

    def __init__(self, mel_bins=128, num_keys=88, cnn_channels=256,
                 lstm_hidden=256, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(mel_bins, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.output  = nn.Linear(lstm_hidden * 2, num_keys)
        self.sigmoid = nn.Sigmoid()

    def forward(self, mel):
        if mel.dim() == 4:
            mel = mel.squeeze(1)
        x = self.cnn(mel)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        logits = self.output(x)
        logits = logits.permute(0, 2, 1)
        return logits, self.sigmoid(logits)

class _SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CNNTransformer(nn.Module):

    def __init__(self, mel_bins=128, num_keys=88, d_model=256,
                 nhead=8, num_layers=2, dim_ff=1024, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(mel_bins, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.pos_enc = _SinusoidalPE(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output  = nn.Conv1d(d_model, num_keys, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, mel):
        if mel.dim() == 4:
            mel = mel.squeeze(1)
        x = self.cnn(mel)
        x = x.permute(0, 2, 1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        logits = self.output(x)
        return logits, self.sigmoid(logits)

_MODELS = {
    'traditional':     TraditionalSP,
    'cnn_bilstm':      CNNBiLSTM,
    'cnn_transformer': CNNTransformer,
}

def get_model(model_type: str, **kwargs) -> nn.Module:
    if model_type not in _MODELS:
        raise ValueError(f"Unknown model '{model_type}'. Choose from: {list(_MODELS)}")
    return _MODELS[model_type](**kwargs)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on: {device}")
    x = torch.randn(2, 1, 128, 256).to(device)
    for name, cls in _MODELS.items():
        m = cls().to(device)
        logits, probs = m(x)
        print(f"  {name:18s}  params={count_parameters(m):>9,}  out={tuple(logits.shape)}")