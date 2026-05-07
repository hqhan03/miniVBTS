"""
Resolution Study: 해상도별 Force Estimation 성능 비교
1632x1080 원본에서 151x100까지 6단계 다운스케일 → 학습/평가 → 비교 그래프 생성
"""
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
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

# --- Configuration ---
DATA_DIR = Path(r"C:\Users\hq\Downloads\20251202_Force_Estimation\20260509_miniVBTS_force_estimation_data")
CSV_DIR = DATA_DIR / "FTsensor_data"
IMG_DIR = DATA_DIR / "Processed_img"
SAVE_DIR = Path(r"./resolution_study_results")
SAVE_DIR.mkdir(exist_ok=True)

LABEL_COLUMNS = ['Force X', 'Force Y', 'Force Z', 'Torque X', 'Torque Y', 'Torque Z']
BATCH_SIZE = 4
NUM_WORKERS = 4
EPOCHS = 20
LEARNING_RATE = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# 해상도 목록: (width, height) - 원본 비율(1632:1080 ≈ 3:2) 유지, 짧은 변 기준
RESOLUTIONS = [
    (1632, 1080),  # 원본
    (1088, 720),   # 720p
    (725, 480),    # 480p
    (544, 360),    # 360p
    (363, 240),    # 240p
    (151, 100),    # 100p
]


# --- Dataset (원본과 동일, 한 번만 로드) ---
class ForceImageDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []
        self.sample_obj_ids = []
        self.sample_trial_ids = []
        self.all_labels = []
        self._load_metadata()

    def _load_metadata(self):
        print("Metadata 스캐닝 중...")
        for obj_id in range(1, 51):
            csv_folder = CSV_DIR / f"{obj_id:02d}"
            img_folder = IMG_DIR / f"{obj_id:02d}"
            if not img_folder.exists():
                continue
            for trial_id in range(1, 10):
                csv_path = csv_folder / f"{trial_id}_frame_synced.csv"
                img_dir = img_folder / str(trial_id)
                if not (csv_path.exists() and img_dir.exists()):
                    continue
                try:
                    df = pd.read_csv(csv_path)
                    img_files = sorted(list(img_dir.glob("*.jpg")))
                    num_frames = min(len(df), len(img_files))
                    if num_frames == 0:
                        continue
                    labels = df.iloc[:num_frames][
                        ['Force X (N)', 'Force Y (N)', 'Force Z (N)',
                         'Torque X (N-m)', 'Torque Y (N-m)', 'Torque Z (N-m)']
                    ].values.astype(np.float32)
                    start_idx = len(self.all_labels)
                    for i in range(num_frames):
                        self.samples.append((str(img_files[i]), start_idx + i))
                        self.sample_obj_ids.append(obj_id)
                        self.sample_trial_ids.append(trial_id)
                    self.all_labels.extend(labels)
                except Exception as e:
                    print(f"  [Warning] Object {obj_id}, Trial {trial_id}: {e}")
        self.all_labels = np.vstack(self.all_labels)
        self.labels = self.all_labels

    def set_scaler(self, scaler):
        self.labels = scaler.transform(self.all_labels).astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"이미지 로드 실패: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, torch.tensor(self.labels[label_idx])


# --- Transform 생성 ---
def make_transforms(width, height):
    """해상도별 train/val transform 생성 (Resize 포함)"""
    train_tf = A.Compose([
        A.Resize(height, width),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(height, width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_tf, val_tf


# --- 모델 평가 ---
def evaluate_model(model, loader, scaler):
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

    metrics = {}
    for i, col in enumerate(LABEL_COLUMNS):
        error = y_pred[:, i] - y_true[:, i]
        metrics[f'{col}_RMSE'] = float(np.sqrt(np.mean(error ** 2)))
        metrics[f'{col}_MAE'] = float(np.mean(np.abs(error)))
        metrics[f'{col}_Bias'] = float(np.mean(error))
        metrics[f'{col}_R2'] = float(r2_score(y_true[:, i], y_pred[:, i]))
        slope, intercept = np.polyfit(y_true[:, i], y_pred[:, i], 1)
        metrics[f'{col}_Slope'] = float(slope)
        metrics[f'{col}_Intercept'] = float(intercept)
    return metrics, y_true, y_pred


# --- Parity Plot 저장 ---
def save_parity_plots(y_true, y_pred, save_path, title_prefix):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title_prefix, fontsize=16)
    axes = axes.flatten()
    for i, col in enumerate(LABEL_COLUMNS):
        axes[i].scatter(y_true[:, i], y_pred[:, i], alpha=0.3, s=10)
        lims = [min(y_true[:, i].min(), y_pred[:, i].min()),
                max(y_true[:, i].max(), y_pred[:, i].max())]
        axes[i].plot(lims, lims, 'r--', linewidth=1.5)
        axes[i].set_xlim(lims)
        axes[i].set_ylim(lims)
        axes[i].set_aspect('equal')
        axes[i].set_title(col, fontsize=14)
        axes[i].set_xlabel('Actual')
        axes[i].set_ylabel('Predicted')
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        axes[i].text(0.05, 0.95, f'$R^2={r2:.3f}$', transform=axes[i].transAxes,
                     va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.5))
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# --- Detailed Metrics CSV 저장 ---
def save_detailed_metrics(metrics, save_path):
    rows = []
    for col in LABEL_COLUMNS:
        rows.append({
            'Axis': col,
            'RMSE': metrics[f'{col}_RMSE'],
            'MAE': metrics[f'{col}_MAE'],
            'Bias': metrics[f'{col}_Bias'],
            'R2': metrics[f'{col}_R2'],
            'Slope': metrics[f'{col}_Slope'],
            'Intercept': metrics[f'{col}_Intercept'],
        })
    pd.DataFrame(rows).to_csv(save_path, index=False)


# --- 시계열 분석 및 저장 ---
def save_timeseries_analysis(model, dataset, obj_test_ids, scaler, ts_dir, num_trials=10):
    print(f"  [Time-series] {num_trials}개 샘플 추출 중...")
    model.eval()
    ts_dir.mkdir(exist_ok=True)

    test_indices = [i for i, obj_id in enumerate(dataset.sample_obj_ids) if obj_id in obj_test_ids]
    test_pairs = sorted(list(set(
        [(dataset.sample_obj_ids[i], dataset.sample_trial_ids[i]) for i in test_indices]
    )))
    selected_pairs = random.sample(test_pairs, min(num_trials, len(test_pairs)))

    for obj_id, trial_id in selected_pairs:
        indices = [i for i, (o, t) in enumerate(zip(dataset.sample_obj_ids, dataset.sample_trial_ids))
                   if o == obj_id and t == trial_id]

        trial_preds, trial_labels = [], []
        trial_loader = DataLoader(Subset(dataset, indices), batch_size=BATCH_SIZE, shuffle=False)
        with torch.no_grad():
            for imgs, labels in trial_loader:
                imgs = imgs.to(DEVICE)
                with torch.amp.autocast(DEVICE.type):
                    preds = model(imgs)
                trial_preds.append(preds.cpu().numpy())
                trial_labels.append(labels.numpy())

        y_pred = scaler.inverse_transform(np.vstack(trial_preds))
        y_true = scaler.inverse_transform(np.vstack(trial_labels))

        # CSV 저장
        df_res = pd.DataFrame()
        df_res['Frame'] = range(len(y_true))
        for i, col in enumerate(LABEL_COLUMNS):
            df_res[f'True_{col}'] = y_true[:, i]
            df_res[f'Pred_{col}'] = y_pred[:, i]
        df_res.to_csv(ts_dir / f"Time-series_Obj{obj_id:02d}_Trial{trial_id}.csv", index=False)

        # Plot
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
        plt.savefig(ts_dir / f"Plot_Obj{obj_id:02d}_Trial{trial_id}.png", dpi=200)
        plt.close()

    print(f"  [Time-series] 완료: {ts_dir}")


# --- 단일 해상도 학습 ---
def train_one_resolution(width, height, full_dataset, train_idx, val_idx,
                         std_test_idx, obj_test_indices, obj_test_ids, scaler):
    label = f"{width}x{height}"
    res_dir = SAVE_DIR / label
    res_dir.mkdir(exist_ok=True)

    train_tf, val_tf = make_transforms(width, height)

    # DataLoaders
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS)
    std_test_loader = DataLoader(Subset(full_dataset, std_test_idx), batch_size=BATCH_SIZE,
                                 num_workers=NUM_WORKERS)
    obj_test_loader = DataLoader(Subset(full_dataset, obj_test_indices), batch_size=BATCH_SIZE,
                                 num_workers=NUM_WORKERS)

    # 매 해상도마다 fresh model
    model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, len(LABEL_COLUMNS))
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    grad_scaler = torch.amp.GradScaler(DEVICE.type)

    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        full_dataset.transform = train_tf
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"[{label}] Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEVICE.type):
                loss = criterion(model(images), labels)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        # --- Validation ---
        model.eval()
        full_dataset.transform = val_tf
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with torch.amp.autocast(DEVICE.type):
                    val_loss += criterion(model(images), labels).item()
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        print(f"[{label}] Epoch {epoch+1}: Train={avg_train:.4f} Val={avg_val:.4f}")

    # Loss history 저장
    pd.DataFrame({
        'epoch': range(1, EPOCHS + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
    }).to_csv(res_dir / "loss_history.csv", index=False)

    # 평가
    full_dataset.transform = val_tf
    std_metrics, std_true, std_pred = evaluate_model(model, std_test_loader, scaler)
    obj_metrics, obj_true, obj_pred = evaluate_model(model, obj_test_loader, scaler)

    # Parity plots
    save_parity_plots(std_true, std_pred, res_dir / "parity_standard_test.png",
                      f"{label} - Standard Test")
    save_parity_plots(obj_true, obj_pred, res_dir / "parity_object_test.png",
                      f"{label} - Object-Based Test")

    # Detailed metrics CSV (Bias, Slope, Intercept 포함)
    save_detailed_metrics(std_metrics, res_dir / "Standard_Test_metrics.csv")
    save_detailed_metrics(obj_metrics, res_dir / "Object_Based_Test_metrics.csv")

    # 시계열 분석 (10 random trials)
    save_timeseries_analysis(model, full_dataset, obj_test_ids, scaler,
                             ts_dir=res_dir / "timeseries_results")

    # 모델 저장
    torch.save(model.state_dict(), res_dir / "model.pth")

    del model
    torch.cuda.empty_cache()

    return std_metrics, obj_metrics


# --- 비교 그래프 생성 ---
def plot_resolution_comparison(results):
    sns.set_theme(style="whitegrid")
    results = sorted(results, key=lambda x: x['short_side'])

    short_sides = [r['short_side'] for r in results]
    res_labels = [r['resolution'] for r in results]

    force_cols = ['Force X', 'Force Y', 'Force Z']
    torque_cols = ['Torque X', 'Torque Y', 'Torque Z']
    colors_f = ['#e41a1c', '#377eb8', '#4daf4a']
    colors_t = ['#ff7f00', '#984ea3', '#a65628']

    # ── 1. R² vs Resolution (Standard / Object 각각) ──
    for test_name, test_key in [("Standard Test", "std_test"), ("Object-Based Test", "obj_test")]:
        fig, ax = plt.subplots(figsize=(10, 6))

        for col, c in zip(force_cols, colors_f):
            vals = [r[test_key][f'{col}_R2'] for r in results]
            ax.plot(short_sides, vals, 'o-', label=col, color=c, lw=2, ms=8)
        for col, c in zip(torque_cols, colors_t):
            vals = [r[test_key][f'{col}_R2'] for r in results]
            ax.plot(short_sides, vals, 's--', label=col, color=c, lw=2, ms=8)

        # Mean R²
        mean_r2 = [np.mean([r[test_key][f'{col}_R2'] for col in LABEL_COLUMNS]) for r in results]
        ax.plot(short_sides, mean_r2, 'D-', label='Mean', color='black', lw=3, ms=10)

        ax.set_xlabel('Short Side Resolution (px)', fontsize=13)
        ax.set_ylabel('R²', fontsize=13)
        ax.set_title(f'R² vs Resolution ({test_name})', fontsize=15)
        ax.set_xticks(short_sides)
        ax.set_xticklabels(res_labels, rotation=45, ha='right')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim(bottom=min(0, min(mean_r2) - 0.1))
        plt.tight_layout()
        plt.savefig(SAVE_DIR / f"R2_vs_resolution_{test_key}.png", dpi=300)
        plt.close()

    # ── 2. RMSE vs Resolution (Force / Torque 분리) ──
    for test_name, test_key in [("Standard Test", "std_test"), ("Object-Based Test", "obj_test")]:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for col, c in zip(force_cols, colors_f):
            vals = [r[test_key][f'{col}_RMSE'] for r in results]
            axes[0].plot(short_sides, vals, 'o-', label=col, color=c, lw=2, ms=8)
        axes[0].set_title(f'Force RMSE ({test_name})', fontsize=14)
        axes[0].set_xlabel('Short Side (px)')
        axes[0].set_ylabel('RMSE (N)')
        axes[0].set_xticks(short_sides)
        axes[0].set_xticklabels(res_labels, rotation=45, ha='right')
        axes[0].legend()

        for col, c in zip(torque_cols, colors_t):
            vals = [r[test_key][f'{col}_RMSE'] for r in results]
            axes[1].plot(short_sides, vals, 's-', label=col, color=c, lw=2, ms=8)
        axes[1].set_title(f'Torque RMSE ({test_name})', fontsize=14)
        axes[1].set_xlabel('Short Side (px)')
        axes[1].set_ylabel('RMSE (N·m)')
        axes[1].set_xticks(short_sides)
        axes[1].set_xticklabels(res_labels, rotation=45, ha='right')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(SAVE_DIR / f"RMSE_vs_resolution_{test_key}.png", dpi=300)
        plt.close()

    # ── 3. Summary: Mean R² + Training Time ──
    fig, ax1 = plt.subplots(figsize=(10, 6))

    mean_r2_std = [np.mean([r['std_test'][f'{c}_R2'] for c in LABEL_COLUMNS]) for r in results]
    mean_r2_obj = [np.mean([r['obj_test'][f'{c}_R2'] for c in LABEL_COLUMNS]) for r in results]
    times_min = [r['elapsed_seconds'] / 60 for r in results]

    ax1.plot(short_sides, mean_r2_std, 'o-', color='#2196F3', lw=2.5, ms=10, label='Standard Test R²')
    ax1.plot(short_sides, mean_r2_obj, 's-', color='#F44336', lw=2.5, ms=10, label='Object Test R²')
    ax1.set_xlabel('Short Side Resolution (px)', fontsize=13)
    ax1.set_ylabel('Mean R²', fontsize=13)
    ax1.set_xticks(short_sides)
    ax1.set_xticklabels(res_labels, rotation=45, ha='right')
    ax1.legend(loc='center left', fontsize=11)

    ax2 = ax1.twinx()
    bar_width = max(15, min(50, (short_sides[-1] - short_sides[0]) / len(short_sides) * 0.3))
    ax2.bar(short_sides, times_min, width=bar_width, alpha=0.2, color='gray', label='Training Time')
    ax2.set_ylabel('Training Time (min)', fontsize=13, color='gray')
    ax2.legend(loc='center right', fontsize=11)

    ax1.set_title('Resolution vs Performance & Training Time', fontsize=15)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "summary_r2_and_time.png", dpi=300)
    plt.close()

    # ── 4. Loss Curves 비교 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for r in results:
        loss_df = pd.read_csv(SAVE_DIR / r['resolution'] / "loss_history.csv")
        axes[0].plot(loss_df['epoch'], loss_df['train_loss'], label=r['resolution'])
        axes[1].plot(loss_df['epoch'], loss_df['val_loss'], label=r['resolution'])

    for ax, title in zip(axes, ['Training Loss', 'Validation Loss']):
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend(fontsize=9)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "loss_curves_comparison.png", dpi=300)
    plt.close()

    print(f"\n비교 그래프 저장 완료: {SAVE_DIR.absolute()}")


# --- Main ---
def load_all_results():
    """저장된 metrics.json 파일들을 모두 로드"""
    results = []
    for w, h in RESOLUTIONS:
        label = f"{w}x{h}"
        result_file = SAVE_DIR / label / "metrics.json"
        if result_file.exists():
            with open(result_file) as f:
                results.append(json.load(f))
        else:
            print(f"  [WARNING] {label} 결과 없음 → 건너뜀")
    return results


def main():
    parser = argparse.ArgumentParser(description="Resolution Study for Force Estimation")
    parser.add_argument('--part', type=int, choices=[1, 2],
                        help="1=고해상도 3개(5090용), 2=저해상도 3개(5070Ti용)")
    parser.add_argument('--merge', action='store_true',
                        help="양쪽 결과 병합 후 비교 그래프만 생성 (GPU 불필요)")
    args = parser.parse_args()

    # --merge 모드: 기존 결과만 로드하여 비교 그래프 생성
    if args.merge:
        print("결과 병합 및 비교 그래프 생성 모드")
        results = load_all_results()
        if len(results) < 2:
            print("비교할 결과가 2개 이상 필요합니다.")
            return
        summary_rows = []
        for r in results:
            row = {'Resolution': r['resolution'], 'Pixels': r['pixels'],
                   'Time(min)': round(r['elapsed_seconds'] / 60, 1)}
            for test_key, prefix in [('std_test', 'Std'), ('obj_test', 'Obj')]:
                for col in LABEL_COLUMNS:
                    row[f'{prefix}_{col}_R2'] = round(r[test_key][f'{col}_R2'], 4)
                    row[f'{prefix}_{col}_RMSE'] = round(r[test_key][f'{col}_RMSE'], 4)
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(SAVE_DIR / "summary.csv", index=False)
        plot_resolution_comparison(results)
        print(f"결과 저장: {SAVE_DIR.absolute()}")
        return

    # GPU 확인 (학습 모드)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾을 수 없습니다. (--merge는 GPU 없이 사용 가능)")

    # Part 분할
    if args.part == 1:
        resolutions = RESOLUTIONS[:3]
        print("Part 1 (5090): 1632x1080, 1088x720, 725x480")
    elif args.part == 2:
        resolutions = RESOLUTIONS[3:]
        print("Part 2 (5070 Ti): 544x360, 363x240, 151x100")
    else:
        resolutions = RESOLUTIONS
        print(f"전체 {len(RESOLUTIONS)}개 해상도 실행")

    total_start = time.time()

    # 1. 데이터셋 로드 (1회)
    full_dataset = ForceImageDataset()

    # 2. 데이터 분할 (원본과 동일한 seed/방식)
    all_obj_ids = list(range(1, 51))
    random.seed(42)
    obj_test_ids = random.sample(all_obj_ids, 5)
    remaining = [i for i in all_obj_ids if i not in obj_test_ids]

    obj_test_indices = [i for i, o in enumerate(full_dataset.sample_obj_ids) if o in obj_test_ids]
    pool_indices = [i for i, o in enumerate(full_dataset.sample_obj_ids) if o in remaining]

    train_idx, temp_idx = train_test_split(pool_indices, test_size=0.4, random_state=42)
    val_idx, std_test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    scaler.fit(full_dataset.all_labels[train_idx])
    full_dataset.set_scaler(scaler)
    joblib.dump(scaler, SAVE_DIR / "scaler.pkl")

    # 3. 해상도별 학습 루프
    results = []

    for i, (w, h) in enumerate(resolutions):
        label = f"{w}x{h}"
        result_file = SAVE_DIR / label / "metrics.json"

        # 이미 완료된 해상도는 건너뜀 (이어하기 지원)
        if result_file.exists():
            print(f"\n[{label}] 이미 완료됨 → 결과 로드")
            with open(result_file) as f:
                results.append(json.load(f))
            continue

        print(f"\n{'='*60}")
        print(f" [{i+1}/{len(resolutions)}] Resolution: {label}")
        print(f"{'='*60}")

        res_start = time.time()
        std_m, obj_m = train_one_resolution(
            w, h, full_dataset, train_idx, val_idx, std_test_idx, obj_test_indices, obj_test_ids, scaler
        )
        elapsed = time.time() - res_start

        # execution_time.csv 저장
        h_e, rem = divmod(int(elapsed), 3600)
        m_e, s_e = divmod(rem, 60)
        pd.DataFrame([{
            "Total_Elapsed_Seconds": round(elapsed, 1),
            "Total_Elapsed": f"{h_e}h {m_e}m {s_e}s",
        }]).to_csv(SAVE_DIR / label / "execution_time.csv", index=False)

        entry = {
            'resolution': label,
            'width': w,
            'height': h,
            'pixels': w * h,
            'short_side': h,
            'elapsed_seconds': round(elapsed, 1),
            'std_test': std_m,
            'obj_test': obj_m,
        }

        (SAVE_DIR / label).mkdir(exist_ok=True)
        with open(result_file, 'w') as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)

        results.append(entry)
        print(f"[{label}] 완료! (소요: {elapsed/60:.1f}분)")

    # 4. Summary CSV
    summary_rows = []
    for r in results:
        row = {
            'Resolution': r['resolution'],
            'Pixels': r['pixels'],
            'Time(min)': round(r['elapsed_seconds'] / 60, 1),
        }
        for test_key, prefix in [('std_test', 'Std'), ('obj_test', 'Obj')]:
            for col in LABEL_COLUMNS:
                row[f'{prefix}_{col}_R2'] = round(r[test_key][f'{col}_R2'], 4)
                row[f'{prefix}_{col}_RMSE'] = round(r[test_key][f'{col}_RMSE'], 4)
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(SAVE_DIR / "summary.csv", index=False)

    # 5. 비교 그래프 생성
    plot_resolution_comparison(results)

    total_elapsed = time.time() - total_start
    h_t, remainder = divmod(int(total_elapsed), 3600)
    m_t, s_t = divmod(remainder, 60)
    print(f"\n총 실행 시간: {h_t}h {m_t}m {s_t}s")
    print(f"결과 저장 위치: {SAVE_DIR.absolute()}")


if __name__ == "__main__":
    main()
