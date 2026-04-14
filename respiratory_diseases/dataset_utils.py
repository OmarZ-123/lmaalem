import os
import re
import random
from collections import defaultdict
from typing import List, Tuple, Dict

import librosa
import numpy as np
import pyloudnorm as pyln
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import GroupShuffleSplit

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def extract_patient_id(path: str) -> str:
    """Robust patient id extractor. Returns 'P<id>' or raises ValueError.

    Tries multiple filename patterns commonly found in recordings.
    """
    fname = os.path.basename(path)
    # pattern: P12 or p12 or patient_12 or patient-12
    m = re.search(r'(?i)(?:p|patient)[-_]?0*(\d+)\b', fname)
    if m:
        return f"P{int(m.group(1))}"
    # fallback: explicit uppercase P followed by digits
    m = re.search(r'P(\d+)', fname)
    if m:
        return f"P{int(m.group(1))}"
    # fallback: digits between separators
    m = re.search(r'[_-](\d{1,4})(?:[_\.-]|$)', fname)
    if m:
        return f"P{int(m.group(1))}"
    raise ValueError(f"No patient id found in filename: {fname}")


def my_add_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise


def my_time_stretch(audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
    return librosa.effects.time_stretch(y=audio, rate=rate)


def my_pitch_shift(audio: np.ndarray, sr: int, n_steps: float = 0.0) -> np.ndarray:
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)


def augment_audio(y: np.ndarray, sr: int):
    choice = random.choice(["noise", "stretch", "pitch", "none"])
    if choice == "noise":
        return my_add_noise(y)
    if choice == "stretch":
        rate = random.uniform(0.85, 1.15)
        try:
            return my_time_stretch(y, rate)
        except Exception:
            return y
    if choice == "pitch":
        steps = random.uniform(-2, 2)
        try:
            return my_pitch_shift(y, sr, n_steps=steps)
        except Exception:
            return y
    return y


def normalize_loudness(y: np.ndarray, sr: int, target_lufs: float = -23.0) -> np.ndarray:
    y = y.astype(np.float64)
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(y)
        y_norm = pyln.normalize.loudness(y, loudness, target_lufs)
        peak = np.max(np.abs(y_norm))
        if peak > 1.0:
            y_norm = y_norm / peak
        return y_norm
    except Exception:
        return y


def save_spectrogram(y: np.ndarray, sr: int, save_path: str, n_mels: int = 128):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(3, 3))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


class SpectrogramDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[str], label_to_idx: Dict[str, int] = None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        if label_to_idx is None:
            unique = sorted(set(labels))
            self.label_to_idx = {lab: i for i, lab in enumerate(unique)}
        else:
            self.label_to_idx = label_to_idx

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        label = self.label_to_idx[self.labels[idx]]
        return img, label


def prepare_train_test_spectrograms(dataset_path: str,
                                    categories: List[str],
                                    spectrogram_dir: str,
                                    test_size: float = 0.2,
                                    random_state: int = 42,
                                    augment: bool = True,
                                    target_lufs: float = -23.0) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, int]]:
    """Prepare patient-wise train/test split, augment train only, normalize and save spectrograms.

    Returns: train_image_paths, train_image_labels, test_image_paths, test_image_labels, label_to_idx
    """
    # 1) collect original wav files
    original_files = []  # (path, label)
    for cat in categories:
        cat_dir = os.path.join(dataset_path, cat)
        if not os.path.isdir(cat_dir):
            continue
        for f in os.listdir(cat_dir):
            if f.lower().endswith('.wav'):
                original_files.append((os.path.join(cat_dir, f), cat))

    if len(original_files) == 0:
        raise RuntimeError('No .wav files found under dataset_path/categories')

    paths = [p for p, _ in original_files]
    labels = [lab for _, lab in original_files]
    groups = [extract_patient_id(p) for p in paths]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(paths, labels, groups))

    train_orig = [original_files[i] for i in train_idx]
    test_orig = [original_files[i] for i in test_idx]

    train_patients = set(extract_patient_id(p) for p, _ in train_orig)
    test_patients = set(extract_patient_id(p) for p, _ in test_orig)
    if len(train_patients & test_patients) != 0:
        raise RuntimeError('Patient leakage detected after split')

    # load original audio for train/test
    train_by_cat = defaultdict(list)
    train_audio_data = []  # (y, sr, label, orig_path)
    for p, lab in tqdm(train_orig, desc='Loading train originals'):
        try:
            y, sr = librosa.load(p, sr=None)
            train_by_cat[lab].append((y, sr, p))
            train_audio_data.append((y, sr, lab, p))
        except Exception as e:
            print('Skip train file:', p, e)

    test_audio_data = []
    for p, lab in tqdm(test_orig, desc='Loading test originals'):
        try:
            y, sr = librosa.load(p, sr=None)
            test_audio_data.append((y, sr, lab, p))
        except Exception as e:
            print('Skip test file:', p, e)

    # ensure at least one sample per class in train
    for cat in categories:
        if len(train_by_cat[cat]) == 0:
            raise RuntimeError(f"No training samples for class '{cat}'")

    # augment training classes to balance
    if augment:
        max_count = max(len(v) for v in train_by_cat.values())
        for cat, items in train_by_cat.items():
            needed = max_count - len(items)
            for _ in range(needed):
                orig_y, orig_sr, orig_path = random.choice(items)
                y_aug = augment_audio(orig_y, orig_sr)
                train_audio_data.append((y_aug, orig_sr, cat, orig_path))

    # normalize audio
    train_norm = []
    for y, sr, lab, orig_path in tqdm(train_audio_data, desc='Normalizing train'):
        y_norm = normalize_loudness(y, sr, target_lufs=target_lufs)
        train_norm.append((y_norm, sr, lab, orig_path))

    test_norm = []
    for y, sr, lab, orig_path in tqdm(test_audio_data, desc='Normalizing test'):
        y_norm = normalize_loudness(y, sr, target_lufs=target_lufs)
        test_norm.append((y_norm, sr, lab, orig_path))

    # save spectrograms
    train_image_paths, train_image_labels = [], []
    test_image_paths, test_image_labels = [], []

    for i, (y, sr, lab, orig_path) in enumerate(tqdm(train_norm, desc='Saving train spectrograms')):
        out_dir = os.path.join(spectrogram_dir, 'train', lab)
        base = os.path.basename(orig_path).replace('.wav', '')
        img_name = f"{base}_train_{i}.png"
        img_path = os.path.join(out_dir, img_name)
        save_spectrogram(y, sr, img_path)
        train_image_paths.append(img_path)
        train_image_labels.append(lab)

    for i, (y, sr, lab, orig_path) in enumerate(tqdm(test_norm, desc='Saving test spectrograms')):
        out_dir = os.path.join(spectrogram_dir, 'test', lab)
        base = os.path.basename(orig_path).replace('.wav', '')
        img_name = f"{base}_test_{i}.png"
        img_path = os.path.join(out_dir, img_name)
        save_spectrogram(y, sr, img_path)
        test_image_paths.append(img_path)
        test_image_labels.append(lab)

    # deterministic mapping
    all_label_set = sorted(set(train_image_labels + test_image_labels))
    label_to_idx = {lab: i for i, lab in enumerate(all_label_set)}

    # save metadata
    meta = {
        'label_to_idx': label_to_idx,
        'train_patients': sorted(train_patients),
        'test_patients': sorted(test_patients)
    }
    torch.save(meta, os.path.join(spectrogram_dir, 'dataset_split_meta.pth'))

    return train_image_paths, train_image_labels, test_image_paths, test_image_labels, label_to_idx


def make_dataloaders(train_image_paths: List[str], train_image_labels: List[str],
                     test_image_paths: List[str], test_image_labels: List[str],
                     label_to_idx: Dict[str, int], batch_size: int = 16, num_workers: int = 4):
    train_dataset = SpectrogramDataset(train_image_paths, train_image_labels, label_to_idx=label_to_idx)
    test_dataset = SpectrogramDataset(test_image_paths, test_image_labels, label_to_idx=label_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, test_loader
