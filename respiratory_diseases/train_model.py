"""Train ViT on prepared spectrograms and save model with metadata.

Usage:
    python train_model.py --dataset_path "C:/Users/omarz/OneDrive/Desktop/p2m/respiratory_diseases/Asthma Detection Dataset Version 2/Asthma Detection Dataset Version 2"

The script will prepare spectrograms (if missing) using `dataset_utils.prepare_train_test_spectrograms`,
create dataloaders, fine-tune a ViT model, and save the best checkpoint including `label_to_idx` metadata.
"""
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import timm

from respiratory_diseases.dataset_utils import prepare_train_test_spectrograms, make_dataloaders


def load_or_prepare(dataset_path, categories, spectrogram_dir, test_size, random_state, augment):
    # If spectrogram metadata exists, load image lists from folders, otherwise prepare
    meta_path = os.path.join(spectrogram_dir, 'dataset_split_meta.pth')
    if os.path.isdir(spectrogram_dir) and os.path.exists(meta_path):
        try:
            meta = torch.load(meta_path)
            label_to_idx = meta.get('label_to_idx')

            train_root = os.path.join(spectrogram_dir, 'train')
            test_root = os.path.join(spectrogram_dir, 'test')
            if os.path.isdir(train_root) and os.path.isdir(test_root):
                train_img_p, train_lbls = [], []
                for lab in sorted(os.listdir(train_root)):
                    lab_dir = os.path.join(train_root, lab)
                    if not os.path.isdir(lab_dir):
                        continue
                    for fn in sorted(os.listdir(lab_dir)):
                        if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                            train_img_p.append(os.path.join(lab_dir, fn))
                            train_lbls.append(lab)

                test_img_p, test_lbls = [], []
                for lab in sorted(os.listdir(test_root)):
                    lab_dir = os.path.join(test_root, lab)
                    if not os.path.isdir(lab_dir):
                        continue
                    for fn in sorted(os.listdir(lab_dir)):
                        if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                            test_img_p.append(os.path.join(lab_dir, fn))
                            test_lbls.append(lab)

                if len(train_img_p) > 0 and len(test_img_p) > 0 and label_to_idx is not None:
                    return train_img_p, train_lbls, test_img_p, test_lbls, label_to_idx, meta
        except Exception:
            pass

    # fallback: prepare data from raw wavs
    os.makedirs(spectrogram_dir, exist_ok=True)
    train_img_p, train_lbls, test_img_p, test_lbls, label_to_idx = prepare_train_test_spectrograms(
        dataset_path, categories, spectrogram_dir=spectrogram_dir, test_size=test_size, random_state=random_state, augment=augment)

    meta_path = os.path.join(spectrogram_dir, 'dataset_split_meta.pth')
    meta = {}
    if os.path.exists(meta_path):
        try:
            meta = torch.load(meta_path)
        except Exception:
            meta = {}

    return train_img_p, train_lbls, test_img_p, test_lbls, label_to_idx, meta


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)

    return running_loss / max(1, total), 100.0 * correct / max(1, total)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outs = model(imgs)
            preds = outs.argmax(1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
    return 100.0 * correct / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--spectrogram_dir', type=str, default=None)
    parser.add_argument('--categories', nargs='+', default=['Bronchial', 'pneumonia', 'asthma', 'healthy', 'copd'])
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--out', type=str, default='model_with_meta.pth')
    args = parser.parse_args()

    spectrogram_dir = args.spectrogram_dir or os.path.join(os.path.dirname(args.dataset_path), 'spectograms')

    print('Preparing or loading spectrogram dataset...')
    train_img_p, train_lbls, test_img_p, test_lbls, label_to_idx, meta = load_or_prepare(
        args.dataset_path, args.categories, spectrogram_dir, args.test_size, args.random_state, args.augment)

    print(f'Train images: {len(train_img_p)}, Test images: {len(test_img_p)}')
    print('Label mapping:', label_to_idx)

    train_loader, test_loader = make_dataloaders(train_img_p, train_lbls, test_img_p, test_lbls, label_to_idx,
                                                batch_size=args.batch_size, num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(label_to_idx)

    print('Building model (ViT) and optimizer...')
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    best_path = 'model_best.pth'

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        t1 = time.time()

        print(f"Epoch {epoch+1}/{args.epochs} — {t1-t0:.1f}s | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        # save best
        if test_acc > best_acc:
            best_acc = test_acc
            ckpt = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'label_to_idx': label_to_idx,
                'meta': meta,
            }
            torch.save(ckpt, best_path)
            print('Saved best model ->', best_path)

    # save final model with metadata
    final_ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': args.epochs - 1,
        'label_to_idx': label_to_idx,
        'meta': meta,
    }
    torch.save(final_ckpt, args.out)
    print('Saved final model ->', args.out)


if __name__ == '__main__':
    main()
