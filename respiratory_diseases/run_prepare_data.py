"""Run data preparation: patient-wise split, augment train only, normalize and save spectrograms.

Usage:
    python run_prepare_data.py

Adjust paths below if needed.
"""
from respiratory_diseases.dataset_utils import prepare_train_test_spectrograms, make_dataloaders

import os

# Defaults copied from your notebook; adjust if your paths differ
DATASET_PATH = r"C:\\Users\\omarz\\OneDrive\\Desktop\\p2m\\respiratory_diseases\\Asthma Detection Dataset Version 2\\Asthma Detection Dataset Version 2"
CATEGORIES = ['Bronchial', 'pneumonia', 'asthma', 'healthy', 'copd']
SPECTROGRAM_DIR = os.path.join(os.path.dirname(DATASET_PATH), 'spectograms')

def main():
    os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
    train_img_p, train_lbls, test_img_p, test_lbls, label_to_idx = prepare_train_test_spectrograms(
        DATASET_PATH, CATEGORIES, spectrogram_dir=SPECTROGRAM_DIR, test_size=0.2, random_state=42, augment=True)

    print('Prepared:', len(train_img_p), 'train images,', len(test_img_p), 'test images')
    print('Label mapping:', label_to_idx)

    # Optionally create dataloaders
    train_loader, test_loader = make_dataloaders(train_img_p, train_lbls, test_img_p, test_lbls, label_to_idx)
    print('Train loader batches:', len(train_loader), 'Test loader batches:', len(test_loader))

if __name__ == '__main__':
    main()
