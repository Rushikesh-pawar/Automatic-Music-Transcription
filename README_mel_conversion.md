# Audio -> Mel conversion

Place your audio files (e.g., `.wav`, `.mp3`) under the `2015/` folder (or another folder) and run the converter to produce mel spectrograms saved as `.npy` files.

Example usage:

```bash
python scripts/convert_to_mel.py --input-dir 2015 --output-dir mels --save-png
```

Default parameters: `sr=22050`, `n_mels=128`, `n_fft=2048`, `hop_length=512`, `fmax=8000`.

Install deps:

```bash
pip install -r requirements.txt
```
