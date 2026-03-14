# Progress Report

What has been completed so far:

- Converted audio to mel spectrograms using `scripts/convert_to_mel.py`.
- Converted `.midi` files to 88-key binary piano-roll matrices aligned with mel frames using `scripts/midi_to_pianoroll.py`.
- Saved paired `.npz` files (mel + piano-roll) and generated `pianorolls/pairs.csv`.
- Added `scripts/eda_visualize.py` to produce sample paired visualizations and a dataset analysis report.

Notes:
- Raw audio and MIDI are under `Audio Files/2015/` in this workspace; scripts default to that location.

Next recommended steps:

- Build a PyTorch Dataset loader that takes mel arrays and piano-roll targets.
- Split data into train/validation/test sets and prepare data augmentation pipelines.
