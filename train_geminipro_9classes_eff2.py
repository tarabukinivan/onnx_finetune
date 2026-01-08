import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np

# ================= КОНФИГУРАЦИЯ =================
DATA_DIR = 'datasets19_57_combi'
CSV_FILE = 'datasets19_57_combi/labels_nv400.csv'
MODEL_SAVE_PATH = 'best_model_b4_clean.pth'

IMG_SIZE = 380
BATCH_SIZE = 16  # Увеличиваем батч (память позволяет, т.к. нет сложных вычислений весов)
EPOCHS = 15      # На сбалансированных данных модель учится быстрее
LR = 3e-4        # Стандартный LR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

TRICORDER_CLASSES = ['AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF', 'INF', 'MEL', 'NV', 'VASC']
CLASS_TO_IDX = {cls: i for i, cls in enumerate(TRICORDER_CLASSES)}

# ================= АУГМЕНТАЦИИ =================
train_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Transpose(p=0.5),
    # Немного геометрии
    A.Affine(scale=(0.8, 1.2), translate_percent=(0.0, 0.1), rotate=(-180, 180), p=0.7),
    # Немного цвета (осторожно)
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ================= ДАТАСЕТ =================
class SkinDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['NewFileName'] if 'NewFileName' in row else row.iloc[0]
        img_path = os.path.join(self.root_dir, img_name)
        label = CLASS_TO_IDX[row['Class']]
        
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except:
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            
        if self.transform:
            image = self.transform(image=image)['image']
        return image, torch.tensor(label, dtype=torch.long)

# ================= ОБУЧЕНИЕ =================
def train():
    torch.manual_seed(42)
    np.random.seed(42)

    print("📊 Подготовка данных (CLEAN MODE)...")
    df = pd.read_csv(CSV_FILE)
    df = df[df['Class'].isin(TRICORDER_CLASSES)]
    
    # Делим на обучение и валидацию
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['Class'], random_state=42)
    
    # === ОБЫЧНЫЙ DATALOADER (БЕЗ SAMPLER) ===
    # shuffle=True сам всё перемешает. Данных поровну, так что всё честно.
    train_loader = DataLoader(SkinDataset(train_df, DATA_DIR, train_transforms), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
                              
    val_loader = DataLoader(SkinDataset(val_df, DATA_DIR, val_transforms), 
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"🚀 Загрузка EfficientNet-B2...")
    model = models.efficientnet_b2(weights='DEFAULT')
    
    # === ПРОСТАЯ ГОЛОВА ===
    # Сложная голова (Hardswish и т.д.) нужна была для борьбы с дисбалансом.
    # Сейчас вернемся к классике, чтобы избежать переобучения.
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, len(TRICORDER_CLASSES))
    )
    model = model.to(DEVICE)
    
    # === ОБЫЧНЫЙ LOSS ===
    # Убрали веса. Добавили label_smoothing, чтобы модель не была слишком самоуверенной.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=2)
    
    best_f1 = 0.0
    print("🔥 Старт обучения (Balanced Data)...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}")
        
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # Валидация
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_f1_macro = f1_score(all_labels, all_preds, average='macro')
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"   Acc: {val_acc:.2%} | Macro F1: {val_f1_macro:.4f} | LR: {current_lr:.2e}")
        
        scheduler.step(val_f1_macro)
        
        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"💾 Saved Best Model (F1: {best_f1:.4f})")

if __name__ == "__main__":
    train()
