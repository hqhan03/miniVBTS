"""
Resolution Study v2: multi-seed + best-val checkpointing + grad clipping + inference latency
- 기존 v1 결과는 seed=42로 옮긴 뒤 seed=[0, 1]만 추가 학습하면 됨
- training time 비교는 폐기, inference latency로 대체
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
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 20
LEARNING_RATE = 1e-4
GRAD_CLIP_MAX_NORM = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # 결정성보다 속도 우선

RESOLUTIONS = [
    (1632, 1080),
    (1088, 720),
    (725, 480),
    (544, 360),
    (363, 240),
    (151, 100),
]
SEEDS = [42, 0, 1]  # 기존 결과 = seed 42; 0, 1 추가로 돌리면 됨


# ============================================================
# Seed utilities
# ============================================================
def set_seed(seed: int):
    """전역 RNG seed (cudnn.benchmark는 그대로 둠)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """DataLoader worker별 seeding"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# Dataset (v1과 동일)
# ============================================================
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


# ============================================================
# Transforms
# ============================================================
def make_transforms(width, height):
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


# ============================================================
# Evaluation
# ============================================================
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

    y_pred = scaler.inverse_transform(np.vstack(all_preds)).astype(np.float64)
    y_true = scaler.inverse_transform(np.vstack(all_truth)).astype(np.float64)

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


# ============================================================
# Inference latency 측정
# ============================================================
def measure_inference_latency(model, height, width, n_warmup=20, n_iter=100):
    """CUDA Event 기반 single-image inference latency (ms)
    배치=1, AMP on. 같은 GPU에서 모든 해상도를 측정해야 fair.
    """
    if not torch.cuda.is_available():
        return None, None

    model.eval()
    dummy = torch.randn(1, 3, height, width, device=DEVICE)

    with torch.no_grad():
        for _ in range(n_warmup):
            with torch.amp.autocast(DEVICE.type):
                _ = model(dummy)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = []
    with torch.no_grad():
        for _ in range(n_iter):
            starter.record()
            with torch.amp.autocast(DEVICE.type):
                _ = model(dummy)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))
    return float(np.median(times)), float(np.percentile(times, 90))


# ============================================================
# Parity / detailed metrics / time-series (v1과 동일)
# ============================================================
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
        axes[i].set_xlim(lims); axes[i].set_ylim(lims); axes[i].set_aspect('equal')
        axes[i].set_title(col, fontsize=14)
        axes[i].set_xlabel('Actual'); axes[i].set_ylabel('Predicted')
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        axes[i].text(0.05, 0.95, f'$R^2={r2:.3f}$', transform=axes[i].transAxes,
                     va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.5))
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()


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


def save_timeseries_analysis(model, dataset, obj_test_ids, scaler, ts_dir, num_trials=10, seed=42):
    print(f"  [Time-series] {num_trials}개 샘플 추출 중...")
    model.eval()
    ts_dir.mkdir(exist_ok=True)

    test_indices = [i for i, obj_id in enumerate(dataset.sample_obj_ids) if obj_id in obj_test_ids]
    test_pairs = sorted(list(set(
        [(dataset.sample_obj_ids[i], dataset.sample_trial_ids[i]) for i in test_indices]
    )))
    rng = random.Random(seed)  # seed별 다른 trial 뽑힘
    selected_pairs = rng.sample(test_pairs, min(num_trials, len(test_pairs)))

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

        y_pred = scaler.inverse_transform(np.vstack(trial_preds)).astype(np.float64)
        y_true = scaler.inverse_transform(np.vstack(trial_labels)).astype(np.float64)

        df_res = pd.DataFrame()
        df_res['Frame'] = range(len(y_true))
        for i, col in enumerate(LABEL_COLUMNS):
            df_res[f'True_{col}'] = y_true[:, i]
            df_res[f'Pred_{col}'] = y_pred[:, i]
        df_res.to_csv(ts_dir / f"Time-series_Obj{obj_id:02d}_Trial{trial_id}.csv", index=False)

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f"Object {obj_id:02d} - Trial {trial_id}", fontsize=18)
        axes = axes.flatten()
        for i, col in enumerate(LABEL_COLUMNS):
            axes[i].plot(df_res['Frame'], df_res[f'True_{col}'], label='Ground Truth', color='black', alpha=0.6)
            axes[i].plot(df_res['Frame'], df_res[f'Pred_{col}'], label='Predicted', color='red', linestyle='--')
            axes[i].set_title(f"{col} Estimation", fontsize=14)
            axes[i].set_xlabel("Frame Index"); axes[i].set_ylabel("Value")
            axes[i].legend(); axes[i].grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(ts_dir / f"Plot_Obj{obj_id:02d}_Trial{trial_id}.png", dpi=200)
        plt.close()


# ============================================================
# Single (resolution, seed) 학습
# ============================================================
def train_one_run(width, height, seed, full_dataset, train_idx, val_idx,
                  std_test_idx, obj_test_indices, obj_test_ids, scaler):
    label = f"{width}x{height}"
    run_dir = SAVE_DIR / label / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    train_tf, val_tf = make_transforms(width, height)

    g = torch.Generator(); g.manual_seed(seed)

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
                              worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS, worker_init_fn=seed_worker)
    std_test_loader = DataLoader(Subset(full_dataset, std_test_idx), batch_size=BATCH_SIZE,
                                 num_workers=NUM_WORKERS, worker_init_fn=seed_worker)
    obj_test_loader = DataLoader(Subset(full_dataset, obj_test_indices), batch_size=BATCH_SIZE,
                                 num_workers=NUM_WORKERS, worker_init_fn=seed_worker)

    model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, len(LABEL_COLUMNS))
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    grad_scaler = torch.amp.GradScaler(DEVICE.type)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None
    best_epoch = -1

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        full_dataset.transform = train_tf
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"[{label} seed={seed}] Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEVICE.type):
                loss = criterion(model(images), labels)
            grad_scaler.scale(loss).backward()
            # ★ clip 전 unscale (AMP 표준 순서)
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        # --- Val ---
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

        # ★ best-val 갱신
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[{label} seed={seed}] Epoch {epoch+1}: Train={avg_train:.4f} Val={avg_val:.4f}"
              f"{' ★ best' if best_epoch == epoch + 1 else ''}")

    # Loss history
    pd.DataFrame({
        'epoch': range(1, EPOCHS + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
    }).to_csv(run_dir / "loss_history.csv", index=False)

    # ★ best checkpoint로 복원 후 평가
    model.load_state_dict(best_state)
    full_dataset.transform = val_tf
    std_metrics, std_true, std_pred = evaluate_model(model, std_test_loader, scaler)
    obj_metrics, obj_true, obj_pred = evaluate_model(model, obj_test_loader, scaler)

    save_parity_plots(std_true, std_pred, run_dir / "parity_standard_test.png",
                      f"{label} seed={seed} - Standard Test")
    save_parity_plots(obj_true, obj_pred, run_dir / "parity_object_test.png",
                      f"{label} seed={seed} - Object-Based Test")
    save_detailed_metrics(std_metrics, run_dir / "Standard_Test_metrics.csv")
    save_detailed_metrics(obj_metrics, run_dir / "Object_Based_Test_metrics.csv")

    save_timeseries_analysis(model, full_dataset, obj_test_ids, scaler,
                             ts_dir=run_dir / "timeseries_results", seed=seed)

    # ★ inference latency 측정 (best checkpoint 기준)
    lat_median, lat_p90 = measure_inference_latency(model, height, width)

    torch.save(best_state, run_dir / "model_best.pth")

    result = {
        'resolution': label, 'width': width, 'height': height,
        'pixels': width * height, 'short_side': height,
        'seed': seed,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'inference_latency_ms_median': lat_median,
        'inference_latency_ms_p90': lat_p90,
        'std_test': std_metrics,
        'obj_test': obj_metrics,
    }
    with open(run_dir / "metrics.json", 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    del model, best_state
    torch.cuda.empty_cache()
    return result


# ============================================================
# Aggregation across seeds
# ============================================================
def aggregate_seeds(resolution_label):
    """한 resolution의 seed별 결과를 mean/std로 집계"""
    res_dir = SAVE_DIR / resolution_label
    seed_results = []
    for seed in SEEDS:
        f = res_dir / f"seed_{seed}" / "metrics.json"
        if f.exists():
            with open(f) as fp:
                seed_results.append(json.load(fp))
    if not seed_results:
        return None

    agg = {
        'resolution': resolution_label,
        'width': seed_results[0]['width'],
        'height': seed_results[0]['height'],
        'short_side': seed_results[0]['short_side'],
        'pixels': seed_results[0]['pixels'],
        'n_seeds': len(seed_results),
        'seeds': [r['seed'] for r in seed_results],
    }

    # Inference latency (seed별 거의 동일하지만 평균)
    lats = [r['inference_latency_ms_median'] for r in seed_results if r.get('inference_latency_ms_median')]
    if lats:
        agg['inference_latency_ms_mean'] = float(np.mean(lats))
        agg['inference_latency_ms_std'] = float(np.std(lats))

    # 모든 metric에 mean/std 계산
    for test_key in ['std_test', 'obj_test']:
        agg[test_key] = {}
        for col in LABEL_COLUMNS:
            for metric in ['R2', 'RMSE', 'MAE', 'Bias']:
                key = f'{col}_{metric}'
                vals = [r[test_key][key] for r in seed_results]
                agg[test_key][f'{key}_mean'] = float(np.mean(vals))
                agg[test_key][f'{key}_std'] = float(np.std(vals))

    with open(res_dir / "aggregated.json", 'w') as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    return agg


# ============================================================
# 비교 그래프 (mean ± std error bar)
# ============================================================
def plot_resolution_comparison(agg_results):
    sns.set_theme(style="whitegrid")
    results = sorted(agg_results, key=lambda x: x['short_side'])
    short_sides = [r['short_side'] for r in results]
    res_labels = [r['resolution'] for r in results]

    force_cols = ['Force X', 'Force Y', 'Force Z']
    torque_cols = ['Torque X', 'Torque Y', 'Torque Z']
    colors_f = ['#e41a1c', '#377eb8', '#4daf4a']
    colors_t = ['#ff7f00', '#984ea3', '#a65628']

    # 1. R² vs Resolution (error bars)
    for test_name, test_key in [("Standard Test", "std_test"), ("Object-Based Test", "obj_test")]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for col, c in zip(force_cols, colors_f):
            means = [r[test_key][f'{col}_R2_mean'] for r in results]
            stds = [r[test_key][f'{col}_R2_std'] for r in results]
            ax.errorbar(short_sides, means, yerr=stds, fmt='o-', label=col, color=c,
                        lw=2, ms=8, capsize=4)
        for col, c in zip(torque_cols, colors_t):
            means = [r[test_key][f'{col}_R2_mean'] for r in results]
            stds = [r[test_key][f'{col}_R2_std'] for r in results]
            ax.errorbar(short_sides, means, yerr=stds, fmt='s--', label=col, color=c,
                        lw=2, ms=8, capsize=4)
        mean_r2 = [np.mean([r[test_key][f'{c}_R2_mean'] for c in LABEL_COLUMNS]) for r in results]
        ax.plot(short_sides, mean_r2, 'D-', label='Mean', color='black', lw=3, ms=10)
        ax.set_xlabel('Short Side Resolution (px)', fontsize=13)
        ax.set_ylabel('R²', fontsize=13)
        ax.set_title(f'R² vs Resolution ({test_name}, mean ± std, n={results[0]["n_seeds"]})', fontsize=15)
        ax.set_xticks(short_sides)
        ax.set_xticklabels(res_labels, rotation=45, ha='right')
        ax.legend(loc='lower right', fontsize=10)
        plt.tight_layout()
        plt.savefig(SAVE_DIR / f"R2_vs_resolution_{test_key}.png", dpi=300)
        plt.close()

    # 2. RMSE vs Resolution
    for test_name, test_key in [("Standard Test", "std_test"), ("Object-Based Test", "obj_test")]:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for col, c in zip(force_cols, colors_f):
            means = [r[test_key][f'{col}_RMSE_mean'] for r in results]
            stds = [r[test_key][f'{col}_RMSE_std'] for r in results]
            axes[0].errorbar(short_sides, means, yerr=stds, fmt='o-', label=col, color=c,
                             lw=2, ms=8, capsize=4)
        axes[0].set_title(f'Force RMSE ({test_name})', fontsize=14)
        axes[0].set_xlabel('Short Side (px)'); axes[0].set_ylabel('RMSE (N)')
        axes[0].set_xticks(short_sides)
        axes[0].set_xticklabels(res_labels, rotation=45, ha='right')
        axes[0].legend()

        for col, c in zip(torque_cols, colors_t):
            means = [r[test_key][f'{col}_RMSE_mean'] for r in results]
            stds = [r[test_key][f'{col}_RMSE_std'] for r in results]
            axes[1].errorbar(short_sides, means, yerr=stds, fmt='s-', label=col, color=c,
                             lw=2, ms=8, capsize=4)
        axes[1].set_title(f'Torque RMSE ({test_name})', fontsize=14)
        axes[1].set_xlabel('Short Side (px)'); axes[1].set_ylabel('RMSE (N·m)')
        axes[1].set_xticks(short_sides)
        axes[1].set_xticklabels(res_labels, rotation=45, ha='right')
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(SAVE_DIR / f"RMSE_vs_resolution_{test_key}.png", dpi=300)
        plt.close()

    # 3. Summary: Mean R² + Inference Latency
    fig, ax1 = plt.subplots(figsize=(10, 6))
    std_means = []
    std_stds = []
    obj_means = []
    obj_stds = []
    for r in results:
        std_per = [r['std_test'][f'{c}_R2_mean'] for c in LABEL_COLUMNS]
        obj_per = [r['obj_test'][f'{c}_R2_mean'] for c in LABEL_COLUMNS]
        std_means.append(np.mean(std_per))
        obj_means.append(np.mean(obj_per))
        # 6-axis 평균값의 seed간 std는 별도 계산이 필요하지만 근사로 평균 std 사용
        std_stds.append(np.mean([r['std_test'][f'{c}_R2_std'] for c in LABEL_COLUMNS]))
        obj_stds.append(np.mean([r['obj_test'][f'{c}_R2_std'] for c in LABEL_COLUMNS]))

    ax1.errorbar(short_sides, std_means, yerr=std_stds, fmt='o-', color='#2196F3',
                 lw=2.5, ms=10, capsize=5, label='Standard Test R²')
    ax1.errorbar(short_sides, obj_means, yerr=obj_stds, fmt='s-', color='#F44336',
                 lw=2.5, ms=10, capsize=5, label='Object Test R²')
    ax1.set_xlabel('Short Side Resolution (px)', fontsize=13)
    ax1.set_ylabel('Mean R²', fontsize=13)
    ax1.set_xticks(short_sides)
    ax1.set_xticklabels(res_labels, rotation=45, ha='right')
    ax1.legend(loc='center left', fontsize=11)

    # Inference latency (ms) bar
    lats = [r.get('inference_latency_ms_mean', 0) for r in results]
    if any(lats):
        ax2 = ax1.twinx()
        bar_width = max(15, min(50, (short_sides[-1] - short_sides[0]) / len(short_sides) * 0.3))
        ax2.bar(short_sides, lats, width=bar_width, alpha=0.2, color='gray', label='Inference Latency')
        ax2.set_ylabel('Inference Latency (ms, batch=1)', fontsize=13, color='gray')
        ax2.legend(loc='center right', fontsize=11)

    ax1.set_title('Resolution vs Performance & Inference Latency', fontsize=15)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "summary_r2_and_latency.png", dpi=300)
    plt.close()

    # 4. Loss curves (seed=42만, 비교용)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for r in results:
        loss_f = SAVE_DIR / r['resolution'] / "seed_42" / "loss_history.csv"
        if not loss_f.exists():
            continue
        loss_df = pd.read_csv(loss_f)
        axes[0].plot(loss_df['epoch'], loss_df['train_loss'], label=r['resolution'])
        axes[1].plot(loss_df['epoch'], loss_df['val_loss'], label=r['resolution'])
    for ax, title in zip(axes, ['Training Loss (seed=42)', 'Validation Loss (seed=42)']):
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
        ax.legend(fontsize=9); ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "loss_curves_comparison.png", dpi=300)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help="실행할 seed들 (e.g., --seeds 0 1). 기본은 SEEDS 전체.")
    parser.add_argument('--resolutions', type=str, nargs='+', default=None,
                        help="실행할 해상도 (e.g., --resolutions 1632x1080). 기본은 전체.")
    parser.add_argument('--merge', action='store_true',
                        help="기존 결과만 aggregate + plot")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else SEEDS
    if args.resolutions is not None:
        resolutions = [(int(s.split('x')[0]), int(s.split('x')[1])) for s in args.resolutions]
    else:
        resolutions = RESOLUTIONS

    # --merge: aggregate만
    if args.merge:
        agg_results = []
        for w, h in RESOLUTIONS:
            label = f"{w}x{h}"
            agg = aggregate_seeds(label)
            if agg is not None:
                agg_results.append(agg)

        # Summary CSV
        summary_rows = []
        for r in agg_results:
            row = {
                'Resolution': r['resolution'],
                'Pixels': r['pixels'],
                'N_seeds': r['n_seeds'],
                'Latency_ms': round(r.get('inference_latency_ms_mean', 0), 3),
            }
            for test_key, prefix in [('std_test', 'Std'), ('obj_test', 'Obj')]:
                for col in LABEL_COLUMNS:
                    row[f'{prefix}_{col}_R2_mean'] = round(r[test_key][f'{col}_R2_mean'], 4)
                    row[f'{prefix}_{col}_R2_std'] = round(r[test_key][f'{col}_R2_std'], 4)
                    row[f'{prefix}_{col}_RMSE_mean'] = round(r[test_key][f'{col}_RMSE_mean'], 4)
                    row[f'{prefix}_{col}_RMSE_std'] = round(r[test_key][f'{col}_RMSE_std'], 4)
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(SAVE_DIR / "summary_aggregated.csv", index=False)

        plot_resolution_comparison(agg_results)
        print(f"Aggregation 완료: {SAVE_DIR.absolute()}")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 필요 (--merge는 GPU 없이 가능)")

    # 데이터셋 + split (seed 무관)
    full_dataset = ForceImageDataset()

    all_obj_ids = list(range(1, 51))
    random.seed(42)  # split은 항상 동일하게 (seed 변경의 영향을 model init/training에만 한정)
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

    # 학습 루프 (resolution × seed)
    total_start = time.time()
    for w, h in resolutions:
        label = f"{w}x{h}"
        for seed in seeds:
            result_file = SAVE_DIR / label / f"seed_{seed}" / "metrics.json"
            if result_file.exists():
                print(f"[{label} seed={seed}] 이미 완료 → skip")
                continue
            print(f"\n{'='*60}\n [{label} seed={seed}]\n{'='*60}")
            run_start = time.time()
            train_one_run(w, h, seed, full_dataset, train_idx, val_idx,
                          std_test_idx, obj_test_indices, obj_test_ids, scaler)
            print(f"[{label} seed={seed}] 완료 ({(time.time()-run_start)/60:.1f}분)")

    # Aggregate + plot
    agg_results = []
    for w, h in RESOLUTIONS:
        agg = aggregate_seeds(f"{w}x{h}")
        if agg is not None:
            agg_results.append(agg)
    plot_resolution_comparison(agg_results)

    h_t, rem = divmod(int(time.time() - total_start), 3600)
    m_t, s_t = divmod(rem, 60)
    print(f"\n총 실행 시간: {h_t}h {m_t}m {s_t}s")


if __name__ == "__main__":
    main()