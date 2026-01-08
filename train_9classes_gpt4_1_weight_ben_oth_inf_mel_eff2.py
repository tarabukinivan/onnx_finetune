import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import f1_score

from tqdm import tqdm
import numpy as np
import random

# ==== CONFIG ====
DATA_CSV = "datasets19_57_combi/labels_nv400.csv"         # <-- Ваш путь к исходному .csv (с колонками Class, NewFileName)
DATA_DIR = "datasets19_57_combi"      # <-- Папка с изображениями
IMG_SIZE = 380
BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-4
SEED = 42
MODEL_SAVE = "best_b4_boost_mel_eff2.pth"

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

ALL_CLASSES = [
    'AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF', 'INF',
    'MAL_OTH', 'MEL', 'NV', 'SCCKA', 'VASC'
]
TRAIN_CLASSES = [c for c in ALL_CLASSES if c not in ['MAL_OTH', 'SCCKA']]
NUM_CLASSES = len(TRAIN_CLASSES)

CLASS_TO_IDX = {c: i for i, c in enumerate(TRAIN_CLASSES)}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# ==== Transforms ====
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.13, scale_limit=0.1, rotate_limit=30, p=0.7),
    A.OneOf([
        A.RandomBrightnessContrast(0.3, 0.3),
        A.HueSaturationValue(10, 25, 10),
    ], p=0.6),
    A.CoarseDropout(max_height=32, max_width=32, p=0.25),
    A.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ToTensorV2()
])

# ==== Dataset ====
class SkinDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fn = row['NewFileName'] if 'NewFileName' in row else row.iloc[0]
        img_path = os.path.join(self.root_dir, fn)
        label = CLASS_TO_IDX[row['Class']]
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except:
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, torch.tensor(label, dtype=torch.long)

# ==== Balance utility ====
def upsample_class(df, cl, n_target):
    df_sub = df[df['Class'] == cl]
    if len(df_sub) == 0:
        return df
    if len(df_sub) >= n_target:
        return df
    more = resample(df_sub, replace=True, n_samples=n_target-len(df_sub), random_state=SEED)
    return pd.concat([df, more])

def balance_df_smart(df, upsample_dict):
    # upsample_dict: {class: target_n}
    for cl, n_target in upsample_dict.items():
        df = upsample_class(df, cl, n_target)
    # Shuffle!
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df

# ==== MAIN ====
def main():
    print("\n📋 Загружаем исходный датасет...")
    df = pd.read_csv(DATA_CSV)
    df = df[df['Class'].isin(TRAIN_CLASSES)]

    print("До балансировки:\n", df['Class'].value_counts())

    # === 1. Балансировка только для малых классов === #
    upsample_scheme = {'BEN_OTH': 200, 'INF': 250}
    df = balance_df_smart(df, upsample_scheme)

    print("После балансировки и миксации:\n", df['Class'].value_counts())

    # === 2. Train/val split (стратиф) === #
    train_df, val_df = train_test_split(
        df, test_size=0.10, stratify=df['Class'], random_state=SEED
    )

    print("Train распределение:\n", train_df['Class'].value_counts())
    print("Val распределение:\n", val_df['Class'].value_counts())

    train_ds = SkinDataset(train_df, DATA_DIR, train_transform)
    val_ds = SkinDataset(val_df, DATA_DIR, val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ==== 3. Class weights (BOOST MEL) ====
    # AKIEC, BCC, BKL, DF, NV, VASC : 1
    # BEN_OTH : 2.0, INF : 1.6, MEL : 2.2 (буст)
    class_weights_dict = {
        'AKIEC': 1.0,
        'BCC': 1.0,
        'BEN_OTH': 2.0,
        'BKL': 1.0,
        'DF': 1.0,
        'INF': 1.6,
        'MEL': 2.2,      # Буст MEL!
        'NV': 1.0,
        'VASC': 1.0
    }
    weights = torch.tensor([class_weights_dict[cl] for cl in TRAIN_CLASSES], dtype=torch.float32)
    print("Class weights:", dict(zip(TRAIN_CLASSES, weights.detach().numpy())))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b2(weights="DEFAULT")
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=4)

    best_f1 = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0, 0, 0
        y_true, y_pred = [], []

        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            loop.set_postfix(loss=loss.item())
        train_acc = correct / total

        # ==== Validation ====
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_y_true, val_y_pred = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_y_true.extend(labels.cpu().numpy())
                val_y_pred.extend(preds.cpu().numpy())
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(val_y_true, val_y_pred, average=None, zero_division=0)
        val_f1_macro = f1_score(val_y_true, val_y_pred, average="macro", zero_division=0)

        print(f" Ep {epoch+1:02d}: Val_acc={val_acc:.3f} | Val_loss={avg_val_loss:.4f} | Val_f1_macro={val_f1_macro:.3f}")
        print("  F1 по классам:", {IDX_TO_CLASS[i]: round(val_f1[i], 3) for i in range(len(val_f1))})

        scheduler.step(val_acc)
        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"  💾 Saved best model. F1_macro: {best_f1:.3f}")

    print("✅ Тренировка завершена")
    print("Лучшая macro-F1:", best_f1)

if __name__ == "__main__":
    main()
