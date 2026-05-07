import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- 1. 하드웨어 및 경로 설정 ---
DATA_DIR = Path(r"C:\Users\hq\Downloads\20251202_Force_Estimation\20260509_miniVBTS_force_estimation_data")
CSV_DIR = DATA_DIR / "FTsensor_data"
IMG_DIR = DATA_DIR / "Processed_img"
SAVE_DIR = Path(r"./research_results_1632x1080")
PLOT_DIR = SAVE_DIR / "plots"
TS_DIR = SAVE_DIR / "timeseries_results" # 시계열 결과 저장 폴더

for d in [SAVE_DIR, PLOT_DIR, TS_DIR]:
    d.mkdir(exist_ok=True)

LABEL_COLUMNS = ['Force X', 'Force Y', 'Force Z', 'Torque X', 'Torque Y', 'Torque Z']
BATCH_SIZE = 4
NUM_WORKERS = 4 # 환경에 따라 조정 (Colab/Local)
EPOCHS = 20
LEARNING_RATE = 1e-4
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU를 찾을 수 없습니다. GPU 환경에서 실행해주세요.")
DEVICE = torch.device("cuda")

torch.backends.cudnn.benchmark = True

# --- 2. Dataset 및 전처리 ---
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class ForceImageDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []
        self.sample_obj_ids = []
        self.sample_trial_ids = [] # Trial 추적용 추가
        self.all_labels = []
        self._load_metadata()

    def _load_metadata(self):
        print("Metadata 스캐닝 중...")
        for obj_id in range(1, 51):
            csv_folder = CSV_DIR / f"{obj_id:02d}"
            img_folder = IMG_DIR / f"{obj_id:02d}"
            if not img_folder.exists(): continue
            for trial_id in range(1, 10):
                csv_path = csv_folder / f"{trial_id}_frame_synced.csv"
                img_dir = img_folder / str(trial_id)
                if not (csv_path.exists() and img_dir.exists()): continue
                try:
                    df = pd.read_csv(csv_path)
                    img_files = sorted(list(img_dir.glob("*.jpg")))
                    num_frames = min(len(df), len(img_files))
                    if num_frames == 0: continue

                    labels = df.iloc[:num_frames][['Force X (N)', 'Force Y (N)', 'Force Z (N)',
                                                   'Torque X (N-m)', 'Torque Y (N-m)', 'Torque Z (N-m)']].values.astype(np.float32)

                    start_idx = len(self.all_labels)
                    for i in range(num_frames):
                        # 각 이미지 샘플이 속한 object와 trial 정보를 저장
                        self.samples.append((str(img_files[i]), start_idx + i))
                        self.sample_obj_ids.append(obj_id)
                        self.sample_trial_ids.append(trial_id)
                    self.all_labels.extend(labels)
                except Exception as e:
                    print(f"  [Warning] Object {obj_id}, Trial {trial_id} 로드 실패: {e}")
                    continue
        self.all_labels = np.vstack(self.all_labels)
        self.labels = self.all_labels

    def set_scaler(self, scaler):
        self.labels = scaler.transform(self.all_labels).astype(np.float32)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"이미지 로드 실패: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform: image = self.transform(image=image)['image']
        return image, torch.tensor(self.labels[label_idx])

# --- 3. 시각화 및 평가 함수 ---
def save_professional_plots(y_true, y_pred, prefix):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    for i, col in enumerate(LABEL_COLUMNS):
        axes[i].scatter(y_true[:, i], y_pred[:, i], alpha=0.3, s=10)
        lims = [min(y_true[:, i].min(), y_pred[:, i].min()),
                max(y_true[:, i].max(), y_pred[:, i].max())]
        axes[i].plot(lims, lims, 'r--', linewidth=1.5, label='y = x')
        axes[i].set_xlim(lims)
        axes[i].set_ylim(lims)
        axes[i].set_aspect('equal')
        axes[i].set_title(f'{col}: Actual vs Predicted', fontsize=14)
        axes[i].set_xlabel('Actual')
        axes[i].set_ylabel('Predicted')
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        axes[i].text(0.05, 0.95, f'$R^2 = {r2:.3f}$', transform=axes[i].transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        axes[i].legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{prefix}_parity_plots.png", dpi=300)
    plt.close()

def detailed_evaluation(model, loader, scaler, desc):
    model.eval()
    all_preds, all_truth = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            with torch.amp.autocast(DEVICE.type):
                preds = model(images)
            all_preds.append(preds.cpu().numpy())
            all_truth.append(labels.numpy())

    y_pred = scaler.inverse_transform(np.vstack(all_preds))
    y_true = scaler.inverse_transform(np.vstack(all_truth))

    metrics = []
    for i, col in enumerate(LABEL_COLUMNS):
        error = y_pred[:, i] - y_true[:, i]  # y=x 기준 오차
        bias = np.mean(error)                 # 평균 편향 (양수=과대예측, 음수=과소예측)
        rmse = np.sqrt(np.mean(error ** 2))   # y=x 기준 RMSE
        mae = np.mean(np.abs(error))          # y=x 기준 MAE
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        # 회귀 기울기/절편 (이상적: slope=1, intercept=0)
        slope, intercept = np.polyfit(y_true[:, i], y_pred[:, i], 1)
        metrics.append([col, rmse, mae, bias, r2, slope, intercept])

    df_metrics = pd.DataFrame(metrics, columns=['Axis', 'RMSE', 'MAE', 'Bias', 'R2', 'Slope', 'Intercept'])
    print(f"\n[{desc} Metrics]\n", df_metrics)
    df_metrics.to_csv(SAVE_DIR / f"{desc}_metrics.csv", index=False)
    save_professional_plots(y_true, y_pred, desc)

# --- 4. 랜덤 10개 Trial 시계열 분석 및 저장 함수 ---
def save_timeseries_analysis(model, dataset, obj_test_ids, scaler, num_trials=10):
    print(f"\n[Time-series Analysis] {num_trials}개 샘플 추출 중...")
    model.eval()

    # Object Test Set 내의 (Object, Trial) 쌍 식별
    test_indices = [i for i, obj_id in enumerate(dataset.sample_obj_ids) if obj_id in obj_test_ids]
    test_pairs = sorted(list(set([(dataset.sample_obj_ids[i], dataset.sample_trial_ids[i]) for i in test_indices])))

    # 랜덤하게 10개 Trial 선택
    selected_pairs = random.sample(test_pairs, min(num_trials, len(test_pairs)))

    for obj_id, trial_id in selected_pairs:
        # 해당 Trial에 속하는 인덱스 추출 (이미지 순서 유지)
        indices = [i for i, (o, t) in enumerate(zip(dataset.sample_obj_ids, dataset.sample_trial_ids))
                   if o == obj_id and t == trial_id]

        trial_preds = []
        trial_labels = []

        # 배치 단위로 추론하여 OOM 방지
        trial_loader = DataLoader(Subset(dataset, indices), batch_size=BATCH_SIZE, shuffle=False)
        with torch.no_grad():
            for imgs, labels in trial_loader:
                imgs = imgs.to(DEVICE)
                with torch.amp.autocast(DEVICE.type):
                    preds = model(imgs)
                trial_preds.append(preds.cpu().numpy())
                trial_labels.append(labels.numpy())

        # Scaling 복원
        y_pred = scaler.inverse_transform(np.vstack(trial_preds))
        y_true = scaler.inverse_transform(np.vstack(trial_labels))

        # 1. CSV 저장
        df_res = pd.DataFrame()
        df_res['Frame'] = range(len(y_true))
        for i, col in enumerate(LABEL_COLUMNS):
            df_res[f'True_{col}'] = y_true[:, i]
            df_res[f'Pred_{col}'] = y_pred[:, i]

        csv_filename = f"Time-series_Obj{obj_id:02d}_Trial{trial_id}.csv"
        df_res.to_csv(TS_DIR / csv_filename, index=False)

        # 2. Time-series Plot
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f"Object {obj_id:02d} - Trial {trial_id}", fontsize=18)
        axes = axes.flatten()

        for i, col in enumerate(LABEL_COLUMNS):
            axes[i].plot(df_res['Frame'], df_res[f'True_{col}'], label='Ground Truth', color='black', alpha=0.6)
            axes[i].plot(df_res['Frame'], df_res[f'Pred_{col}'], label='Predicted', color='red', linestyle='--')
            axes[i].set_title(f"{col} Estimation", fontsize=14)
            axes[i].set_xlabel("Frame Index")
            axes[i].set_ylabel("Value")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(TS_DIR / f"Plot_Obj{obj_id:02d}_Trial{trial_id}.png", dpi=200)
        plt.close()

    print(f"시계열 분석 완료: {TS_DIR.absolute()}")

# --- 5. Main Execution ---
def main():
    start_time = time.time()

    # 데이터 로드 및 분할
    full_dataset = ForceImageDataset()
    all_obj_ids = list(range(1, 51))
    random.seed(42)
    obj_test_ids = random.sample(all_obj_ids, 5) # 5개 물체를 아예 테스트용으로 격리
    remaining_obj_ids = [i for i in all_obj_ids if i not in obj_test_ids]

    obj_test_indices = [i for i, obj_id in enumerate(full_dataset.sample_obj_ids) if obj_id in obj_test_ids]
    pool_indices = [i for i, obj_id in enumerate(full_dataset.sample_obj_ids) if obj_id in remaining_obj_ids]

    train_idx, temp_idx = train_test_split(pool_indices, test_size=0.4, random_state=42)
    val_idx, std_test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    scaler.fit(full_dataset.all_labels[train_idx])
    full_dataset.set_scaler(scaler)
    joblib.dump(scaler, SAVE_DIR / "scaler.pkl")

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    std_test_loader = DataLoader(Subset(full_dataset, std_test_idx), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    obj_test_loader = DataLoader(Subset(full_dataset, obj_test_indices), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # 모델 설정
    model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, len(LABEL_COLUMNS))
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    grad_scaler = torch.amp.GradScaler(DEVICE.type)

    # 학습
    print("Training phase starting...")
    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        full_dataset.transform = train_transform

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0

        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(DEVICE.type):
                loss = criterion(model(images), labels)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        full_dataset.transform = val_transform
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with torch.amp.autocast(DEVICE.type):
                    val_loss += criterion(model(images), labels).item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1} 완료. Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 평가
    full_dataset.transform = val_transform
    detailed_evaluation(model, std_test_loader, scaler, "Standard_Test")
    detailed_evaluation(model, obj_test_loader, scaler, "Object_Based_Test")

    # 6. 시계열 결과 별도 저장 (10개 trial)
    save_timeseries_analysis(model, full_dataset, obj_test_ids, scaler, num_trials=10)

    torch.save(model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict(),
               SAVE_DIR / "final_model.pth")

    elapsed = time.time() - start_time
    h, remainder = divmod(int(elapsed), 3600)
    m, s = divmod(remainder, 60)
    time_str = f"{h}h {m}m {s}s"
    print(f"\n총 실행 시간: {time_str}")
    pd.DataFrame([{"Total_Elapsed_Seconds": round(elapsed, 1), "Total_Elapsed": time_str}]).to_csv(
        SAVE_DIR / "execution_time.csv", index=False)

    print(f"모든 결과가 다음 경로에 저장되었습니다: {SAVE_DIR.absolute()}")

if __name__ == "__main__":
    main()
