"""
Models for frame-level piano transcription.

Input:  mel spectrogram  (batch, 1, 128, time)  or  (batch, 128, time)
Output: piano-roll logits/probs  (batch, 88, time)

Models
------
1. TraditionalSP      – Traditional signal processing baseline (NMF / spectral
                        peak-picking). No learned parameters. Plug-in interface
                        for the signal-processing component contributed by a
                        collaborator; stub is in place, implementation pending.
2. CNNBiLSTM          – CNN encoder + 2-layer Bidirectional LSTM
3. CNNTransformer     – CNN encoder + 2-layer Transformer (self-attention)

Comparison
----------
| Model           | Params  | Temporal context      | Speed  | Strength                       |
|-----------------|---------|-----------------------|--------|--------------------------------|
| TraditionalSP   |   0     | Frame-by-frame        | Fast   | No training data needed;       |
|                 |         | (no learned state)    |        | interpretable; strong physics  |
|                 |         |                       |        | prior on harmonic structure    |
| CNNBiLSTM       | ~3.1 M  | Sequential (all past  | Medium | Captures attack/sustain/release|
|                 |         |   + all future via    |        | patterns; handles long notes   |
|                 |         |   bidirectional LSTM) |        |                                |
| CNNTransformer  | ~2.0 M  | Global (any-to-any    | Medium | Long-range harmonic structure; |
|                 |         |   attention)          |        | repeated chord patterns        |

Why CNNBiLSTM > TraditionalSP:
  Traditional methods operate frame-by-frame with hand-crafted harmonic templates.
  They cannot adapt to recording conditions, instrument timbre variations, or
  polyphonic interference without manual tuning. The BiLSTM learns all of these
  from data and also captures temporal note context (attack/sustain/release).

Why CNNTransformer >= CNNBiLSTM:
  Self-attention directly compares every frame to every other frame in O(1) hops,
  without the sequential bottleneck of LSTM. Repeated chord patterns and long-range
  harmonic dependencies are easier to learn via attention weights than through LSTM
  memory gates. The Transformer also achieves this with fewer parameters (2.0M vs 3.1M).
  Trade-off: O(T²) memory for the attention matrix.
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model 1: TraditionalSP  (Traditional Signal Processing — stub)
# ---------------------------------------------------------------------------

class TraditionalSP(nn.Module):
    """
    Traditional signal-processing baseline for piano transcription.

    Approach: frame-by-frame harmonic analysis of the mel spectrogram without
    any learned parameters.  Two standard techniques are combined:

      1. Harmonic Product Spectrum (HPS) — multiplies the spectrum with
         downsampled copies of itself so that harmonic overtones reinforce
         the fundamental frequency, making note peaks easier to detect.

      2. Per-key energy thresholding — maps each of the 88 MIDI piano pitches
         to its nearest mel bin, reads the HPS energy at that bin, and applies
         a percentile-based threshold to produce a binary activation.

    Advantages over learned models:
      • Zero training data required — works out of the box.
      • Fully interpretable: the activation of each key is determined by
        a simple energy comparison, not a black-box weight matrix.
      • Strong physics prior: harmonics of real piano notes are well modelled
        by integer-multiple frequency relationships.

    Limitations:
      • Cannot adapt to recording conditions or timbral variation.
      • Polyphony causes harmonic overlap, increasing false positives.
      • No temporal modelling — each frame is classified independently.

    NOTE: This class provides the interface and a working reference
    implementation.  The full signal-processing pipeline (contributed
    separately) will replace / extend the body of `_sp_transcribe` once
    integrated.
    """

    # MIDI pitch → frequency (Hz)
    @staticmethod
    def _midi_to_freq(pitch: int) -> float:
        return 440.0 * (2.0 ** ((pitch - 69) / 12.0))

    def __init__(self, mel_bins: int = 128, num_keys: int = 88,
                 sr: int = 22050, fmax: float = 8000.0,
                 hps_harmonics: int = 3, threshold_percentile: float = 85.0):
        """
        Parameters
        ----------
        mel_bins            : number of mel frequency bins (must match training data)
        num_keys            : number of piano keys (88)
        sr                  : sample rate used during mel conversion
        fmax                : maximum frequency used during mel conversion
        hps_harmonics       : number of harmonics to multiply in HPS
        threshold_percentile: per-frame percentile above which a key is active
        """
        super().__init__()
        self.mel_bins   = mel_bins
        self.num_keys   = num_keys
        self.hps_harmonics = hps_harmonics
        self.threshold_percentile = threshold_percentile

        # Pre-compute mel-bin index for each of the 88 piano keys
        import librosa
        mel_freqs = librosa.mel_frequencies(n_mels=mel_bins, fmax=fmax)
        key_bins  = []
        for k in range(num_keys):
            freq    = self._midi_to_freq(k + 21)   # A0 = MIDI 21
            bin_idx = int(np.argmin(np.abs(mel_freqs - freq)))
            key_bins.append(min(bin_idx, mel_bins - 1))
        # register as buffer so it moves with .to(device) calls
        self.register_buffer('key_bins',
                             torch.tensor(key_bins, dtype=torch.long))

    # ------------------------------------------------------------------
    # Core signal-processing logic  (replace / extend this method)
    # ------------------------------------------------------------------
    def _sp_transcribe(self, mel_frame: np.ndarray) -> np.ndarray:
        """
        Transcribe a single mel frame (mel_bins,) → activations (88,).

        Current implementation: Harmonic Product Spectrum + percentile threshold.
        Replace with the full traditional pipeline when available.
        """
        spec = np.exp(mel_frame)            # undo log

        # Harmonic Product Spectrum
        hps = spec.copy()
        for h in range(2, self.hps_harmonics + 1):
            downsampled = spec[::h][:len(hps)]
            hps[:len(downsampled)] *= downsampled

        # Read energy at each key's mel bin
        key_bins  = self.key_bins.cpu().numpy()
        energies  = hps[key_bins]

        # Threshold
        thresh    = np.percentile(energies, self.threshold_percentile)
        return (energies >= thresh).astype(np.float32)

    # ------------------------------------------------------------------
    # nn.Module interface  (keeps API identical to the learned models)
    # ------------------------------------------------------------------
    def forward(self, mel: torch.Tensor):
        """
        Parameters
        ----------
        mel : (batch, 1, mel_bins, time)  or  (batch, mel_bins, time)

        Returns
        -------
        logits : (batch, 88, time)   — raw activations (0 or 1 here)
        probs  : (batch, 88, time)   — same (no sigmoid needed for binary output)
        """
        if mel.dim() == 4:
            mel = mel.squeeze(1)                    # (B, mel_bins, T)

        B, M, T   = mel.shape
        mel_np    = mel.cpu().numpy()
        out       = np.zeros((B, self.num_keys, T), dtype=np.float32)

        for b in range(B):
            for t in range(T):
                out[b, :, t] = self._sp_transcribe(mel_np[b, :, t])

        probs  = torch.from_numpy(out).to(mel.device)
        return probs, probs                         # logits == probs for binary


# ---------------------------------------------------------------------------
# Model 2: CNNBiLSTM
# ---------------------------------------------------------------------------

class CNNBiLSTM(nn.Module):
    """
    CNN feature extractor followed by a 2-layer Bidirectional LSTM.

    Architecture:
        CNN      : Conv1d(128→256, k=3) → Conv1d(256→256, k=5)  [no pooling]
        BiLSTM   : 2 layers, hidden=256, bidirectional  →  output dim 512
        Output   : Linear(512→88) per frame

    Advantages over BaselineCNN:
      • The LSTM maintains a hidden state that propagates information across
        the entire time sequence — no fixed receptive field limit.
      • Bidirectionality means each frame receives context from both the past
        (forward LSTM) and the future (backward LSTM), which helps detect
        note onsets (future confirms what the attack started) and offsets.
      • Naturally models the temporal lifecycle of a note: attack → sustain →
        release, as the gate mechanisms learn to open/close accordingly.

    Trade-off: sequential computation prevents full parallelism over time.
    """

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
        self.output = nn.Linear(lstm_hidden * 2, num_keys)
        self.sigmoid = nn.Sigmoid()

    def forward(self, mel):
        if mel.dim() == 4:
            mel = mel.squeeze(1)
        x = self.cnn(mel)                   # (B, C, T)
        x = x.permute(0, 2, 1)             # (B, T, C)
        x, _ = self.lstm(x)                 # (B, T, 2*H)
        logits = self.output(x)             # (B, T, 88)
        logits = logits.permute(0, 2, 1)    # (B, 88, T)
        return logits, self.sigmoid(logits)


# ---------------------------------------------------------------------------
# Model 3: CNNTransformer
# ---------------------------------------------------------------------------

class _SinusoidalPE(nn.Module):
    """Add sinusoidal positional encoding (fixed, not learned)."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        return x + self.pe[:, :x.size(1)]


class CNNTransformer(nn.Module):
    """
    CNN feature extractor followed by a 2-layer Transformer encoder
    (multi-head self-attention + feed-forward).

    Architecture:
        CNN          : Conv1d(128→256, k=3) → Conv1d(256→256, k=5)  [no pooling]
        Positional   : sinusoidal encoding
        Transformer  : 2 × (MultiheadAttention(8 heads) + FFN(256→1024→256))
        Output       : Conv1d(256→88, k=1)

    Advantages over CNNBiLSTM:
      • Self-attention connects ANY two time frames in a single operation —
        no matter how far apart they are. This allows the model to directly
        learn that "if a chord appears at frame 10, it likely reappears at
        frame 500" — something an LSTM must remember across hundreds of steps.
      • Fully parallelisable over time (no sequential dependency), so it
        trains faster on GPU and scales better with longer contexts.
      • Attention maps are more interpretable: each head can specialise in
        e.g. detecting repeated motifs, harmonic intervals, or rhythm patterns.

    Trade-off: attention is O(T²) in memory; very long sequences need chunking
    or sparse attention. For our 256-frame chunks this is not an issue.
    """

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
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)
        self.output = nn.Conv1d(d_model, num_keys, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, mel):
        if mel.dim() == 4:
            mel = mel.squeeze(1)
        x = self.cnn(mel)               # (B, d_model, T)
        x = x.permute(0, 2, 1)         # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)         # (B, T, d_model)
        x = x.permute(0, 2, 1)         # (B, d_model, T)
        logits = self.output(x)         # (B, 88, T)
        return logits, self.sigmoid(logits)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_MODELS = {
    'traditional': TraditionalSP,
    'cnn_bilstm': CNNBiLSTM,
    'cnn_transformer': CNNTransformer,
}


def get_model(model_type: str, **kwargs) -> nn.Module:
    if model_type not in _MODELS:
        raise ValueError(f"Unknown model '{model_type}'. "
                         f"Choose from: {list(_MODELS)}")
    return _MODELS[model_type](**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    x = torch.randn(2, 1, 128, 256)
    for name, cls in _MODELS.items():
        m = cls()
        logits, probs = m(x)
        print(f"{name:18s}  params={count_parameters(m):>9,}  "
              f"out={tuple(logits.shape)}")
