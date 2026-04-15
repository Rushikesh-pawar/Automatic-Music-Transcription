# Progress Report

Project Status Report
What we have achieved: Implemented and tested preprocessing and EDA pipelines, audio to mel spectrograms, MIDI to 88-key piano roll alignment, and paired visualizations. Generated outputs (mels, piano rolls, EDA visualizations) and updated documentation and requirements.txt. Initialized the repository and pushed changes to GitHub.

Immediate next steps:
1.	Migrate the 120GB processed dataset off the repository to free cloud storage (Google Drive or Kaggle Datasets) and keep only small sample files in git for testing
2.	Update the repo to load data remotely via download scripts or mounted drives
3.	Implement a PyTorch Dataset class that streams mel + piano roll pairs on-the-fly for training
   
Upload to Kaggle Datasets or Hugging Face both gives free GPU access and direct dataset mounting during training, which eliminates the need to download 120GB every session.

Updated one month plan:
Week 1: Upload dataset to Kaggle or Hugging Face. Remove large binaries from git, add .gitignore rules, and write a data loading script that mounts/streams directly from the cloud platform.

Week 2: Build and test a PyTorch Dataset class that loads mel + piano roll pairs on-the-fly. Add unit tests and verify alignment on a small sample before scaling up.

Week 3: Train the CNN model on the full dataset using free GPU (Kaggle or Google Colab). Collect evaluation metrics — precision, recall, and F1 per note and overall.

Week 4: Analyze accuracy across note ranges (bass vs treble), identify failure modes, iterate on model architecture or preprocessing, and finalize a demo notebook for reproducibility.

