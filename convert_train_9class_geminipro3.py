import torch
import torch.nn as nn
from torchvision import models
import onnx
import onnxruntime as ort
import numpy as np
import os
import shutil

# --- НАСТРОЙКИ ---
MODEL_PATH = 'best_model_b4_clean.pth' 
ONNX_PATH = 'best_model_b4_clean.onnx'
IMG_SIZE = 380
NUM_CLASSES_TRAINED = 9 

# ================= АРХИТЕКТУРА (Clean / Simple Head) =================
class SafeEfficientNetB0(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        original = models.efficientnet_b4(weights=None) 
        self.features = original.features
        self.avgpool = original.avgpool
        self.classifier = original.classifier
        
        # Простая голова (как в train_clean.py)
        num_ftrs = self.classifier[1].in_features
        self.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ================= ОБЕРТКА (Clean / No Boost) =================
class MiningWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, pixel_values, age_sex_loc):
        # 1. Нормализация
        mean = torch.tensor([0.485, 0.456, 0.406], device=pixel_values.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pixel_values.device).view(1, 3, 1, 1)
        x = (pixel_values - mean) / std

        # 2. Инференс
        logits = self.model(x)
        
        # ЧИСТО: Без буста (+0.0)
        probs = torch.softmax(logits, dim=1) 

        # 3. Вставка классов (до 11)
        batch_size = probs.shape[0]
        zeros = torch.zeros((batch_size, 1), device=pixel_values.device)

        part1 = probs[:, 0:6]
        part2 = probs[:, 6:8]
        part3 = probs[:, 8:]
        
        final_probs = torch.cat([part1, zeros, part2, zeros, part3], dim=1)
        return final_probs

# ================= ЭКСПОРТ И ПРОВЕРКИ =================
def export_and_verify():
    print(f"📂 Загрузка весов из {MODEL_PATH}...")
    device = torch.device('cpu')
    base_model = SafeEfficientNetB0(num_classes=NUM_CLASSES_TRAINED).to(device)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if isinstance(checkpoint, dict):
            st = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        else:
            st = checkpoint
        base_model.load_state_dict(st)
        print("✅ Веса загружены.")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return

    base_model.eval()
    full_model = MiningWrapper(base_model).to(device)
    
    dummy_img = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    dummy_meta = torch.randn(1, 3)

    print(f"📦 Экспорт во временный файл...")
    temp_onnx = "temp_export.onnx"
    if os.path.exists(temp_onnx): os.remove(temp_onnx)

    torch.onnx.export(
        full_model,
        (dummy_img, dummy_meta),
        temp_onnx,
        input_names=['pixel_values', 'age_sex_loc'],
        output_names=['probabilities'],
        opset_version=14,
        dynamic_axes={'pixel_values': {0: 'batch'}, 'age_sex_loc': {0: 'batch'}, 'probabilities': {0: 'batch'}}
    )

    print("🛠  Склейка в один файл...")
    model_proto = onnx.load(temp_onnx)
    
    if os.path.exists(ONNX_PATH): os.remove(ONNX_PATH)
    if os.path.exists(ONNX_PATH + ".data"): os.remove(ONNX_PATH + ".data")
    
    # Сохраняем без внешних данных
    onnx.save_model(model_proto, ONNX_PATH, save_as_external_data=False)
    
    # Чистка
    if os.path.exists(temp_onnx): os.remove(temp_onnx)
    if os.path.exists(temp_onnx + ".data"): os.remove(temp_onnx + ".data")
    
    print(f"✅ Готово: {ONNX_PATH}")
    print(f"   Размер файла: {os.path.getsize(ONNX_PATH) / 1024**2:.2f} MB")

    # ================= ПРОВЕРКИ (VALIDATION) =================
    print("\n🔍 ЗАПУСК ПРОВЕРОК...")
    
    try:
        sess = ort.InferenceSession(ONNX_PATH)
        
        # --- ТЕСТ 1: ИДЕНТИЧНОСТЬ (PyTorch vs ONNX) ---
        print("\n1️⃣  Тест идентичности (Batch=1)...")
        with torch.no_grad():
            torch_out = full_model(dummy_img, dummy_meta).numpy()
        
        ort_out = sess.run(None, {
            'pixel_values': dummy_img.numpy(), 
            'age_sex_loc': dummy_meta.numpy()
        })[0]
        
        diff = np.max(np.abs(torch_out - ort_out))
        print(f"   Максимальная разница значений: {diff:.2e}")
        
        if diff < 1e-4:
            print("   ✅ УСПЕШНО: PyTorch и ONNX выдают одинаковые числа.")
        else:
            print("   ⚠️ ВНИМАНИЕ: Есть расхождения (возможно, из-за оптимизаций).")

        # --- ТЕСТ 2: БАТЧ 50 ---
        print("\n2️⃣  Тест на батче 50 (Проверка динамических осей)...")
        batch_size = 50
        big_img = np.random.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
        big_meta = np.random.randn(batch_size, 3).astype(np.float32)
        
        out_batch = sess.run(None, {
            'pixel_values': big_img, 
            'age_sex_loc': big_meta
        })[0]
        
        print(f"   Вход: (50, 3, {IMG_SIZE}, {IMG_SIZE})")
        print(f"   Выход: {out_batch.shape}")
        
        if out_batch.shape == (50, 11):
            print("   ✅ УСПЕШНО: Размер выхода верный (50, 11).")
        else:
            print(f"   ❌ ОШИБКА: Неверный размер выхода!")

        # Проверка нулевых колонок (MAL_OTH idx=6, SCCKA idx=9)
        col6_sum = np.sum(out_batch[:, 6])
        col9_sum = np.sum(out_batch[:, 9])
        
        print(f"   Сумма MAL_OTH (должно быть 0): {col6_sum}")
        print(f"   Сумма SCCKA   (должно быть 0): {col9_sum}")
        
        if col6_sum == 0 and col9_sum == 0:
            print("   ✅ УСПЕШНО: Пустые классы корректно занулены.")
        else:
            print("   ❌ ОШИБКА: Пустые классы содержат значения!")

    except Exception as e:
        print(f"❌ FATAL ERROR во время тестов: {e}")

if __name__ == "__main__":
    export_and_verify()
