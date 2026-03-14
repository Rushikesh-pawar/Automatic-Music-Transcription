# MAESTRO v3 — Dataset Notes

We are using the MAESTRO v3 dataset (MIDI + aligned audio) as the ground-truth target for Automatic Music Transcription (AMT) for our project.

Key points:
- MAESTRO v3 contains paired waveform (.wav) and MIDI (.midi/.mid) files recorded from Yamaha Disklavier pianos and aligned. It is commonly used for supervised AMT research.
- The dataset contains about 200 hours of paired audio and MIDI recordings from ten years of International Piano-e-Competition
- It is 120 GB dataset but we are using a subset in beginning to check if everything is working well. 
- For this project we reference the dataset as provided (audio + MIDI). We only use the audio to create mel-spectrogram inputs and the MIDI to create 88-key piano-roll targets.
 - For this project we reference the dataset as provided (audio + MIDI). Place the raw files under `Audio Files/2015/` — the preprocessing scripts expect audio and MIDI in that folder by default. We use the audio to create mel-spectrogram inputs and the MIDI to create 88-key piano-roll targets.

Piano-roll format used here:
- 88 rows corresponding to MIDI pitches 21..108 (A0..C8) — standard piano range.
- Binary values per time frame: 1 if a note is active in that frame, 0 otherwise.
- Time frames are aligned to mel spectrogram frames (STFT hop-length determines frame rate). We trim/pad piano-rolls to exactly match mel frame counts.

Files created by preprocessing scripts:
- `mels/` — mel spectrogram .npy arrays (and optional PNG previews).
- `pianorolls/` — 88-key piano-roll `.npy` arrays and `.npz` paired files when mel available.
- `pianorolls/pairs.csv` — CSV listing MIDI, piano-roll, and mel file paths.

