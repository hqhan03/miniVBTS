# miniVBTS Force Estimation

Vision-Based Tactile Sensor (VBTS)의 영상 데이터로부터 6축 힘/토크(Force/Torque)를 추정하는 딥러닝 파이프라인입니다.

## Overview

miniVBTS로 촬영된 접촉 영상에서 프레임 단위 밝기 변화를 추출하고, DenseNet161 기반 회귀 모델을 학습하여 6축 힘/토크 값(Fx, Fy, Fz, Tx, Ty, Tz)을 예측합니다.

## Pipeline

### 1. Video Preprocessing
영상의 각 프레임과 기준 프레임(첫 5프레임 평균) 간 밝기 차이를 컬러로 시각화합니다.
- **Green**: 밝아진 영역 (압축)
- **Red**: 어두워진 영역 (인장)
- **Blue**: 기준 프레임 밝기

| Script | Description |
|---|---|
| `vid_process_RGB.py` | 단일 영상 처리 (CLI) |
| `vid_process_RG` | 단일 영상 처리 (Blue 채널 제거 버전) |
| `vid_process_RGB_img_for_all.py` | 전체 데이터셋 일괄 처리 (멀티프로세싱) |

### 2. FT Sensor Data Synchronization
영상 프레임과 FT 센서 데이터를 시간 동기화하여 프레임별 라벨(Force/Torque)을 생성합니다.

| Script | Description |
|---|---|
| `FTdata_sync.py` | 단일 영상-센서 동기화 |
| `FTdata_sync_for_all.py` | 전체 데이터셋 일괄 동기화 |

### 3. Force Estimation Model
DenseNet161을 fine-tuning하여 이미지 -> 6축 Force/Torque 회귀 모델을 학습합니다.

| Script | Description |
|---|---|
| `force_estimation.py` | 224x224 리사이즈 버전 (원본 해상도 무관) |
| `force_estimation_1632x1080.py` | 1632x1080 원본 해상도 입력 버전 |
| `resolution_study.py` | 해상도별 성능 비교 실험 (1632x1080 ~ 151x100) |

**Model Details:**
- Backbone: DenseNet161 (pretrained on ImageNet)
- Output: 6-axis Force/Torque (Fx, Fy, Fz, Tx, Ty, Tz)
- Loss: MSE
- Optimizer: AdamW
- Mixed Precision Training (AMP)
- Data Split: Object-based test set (5 unseen objects) + Standard random split

**Evaluation:**
- Parity plots (Actual vs Predicted)
- Metrics: RMSE, MAE, R2, Bias, Slope, Intercept
- Time-series analysis (10 random trials)

### 4. Resolution Study
입력 해상도가 Force/Torque 추정 정확도에 미치는 영향을 비교 분석합니다.

**테스트 해상도 (원본 비율 3:2 유지):**

| Label | Resolution | Pixels |
|---|---|---|
| 1080p | 1632x1080 | 1.76M |
| 720p | 1088x720 | 783K |
| 480p | 725x480 | 348K |
| 360p | 544x360 | 196K |
| 240p | 363x240 | 87K |
| 100p | 151x100 | 15K |

**2-GPU 분산 실행 지원:**
```bash
# 5090 (고해상도 3개)
python resolution_study.py --part 1

# 5070 Ti (저해상도 3개)
python resolution_study.py --part 2

# 양쪽 결과 병합 후 비교 그래프 생성 (GPU 불필요)
python resolution_study.py --merge
```

**해상도별 출력 (force_estimation_1632x1080.py와 동일):**
- Detailed metrics CSV (RMSE, MAE, Bias, R², Slope, Intercept)
- Parity plots (Standard Test / Object-Based Test)
- Time-series analysis (10 random trials - 시계열 플롯 + CSV)
- Loss history, 모델 파일, execution_time.csv

**비교 그래프 (--merge):**
- 해상도별 R², RMSE 비교 그래프
- Loss curves 비교
- 해상도-성능-학습시간 종합 Summary

## Requirements

- Python 3.8+
- PyTorch (CUDA)
- torchvision
- OpenCV
- albumentations
- scikit-learn
- pandas, numpy, matplotlib, seaborn
- joblib, tqdm

## Project Structure

```
├── force_estimation.py              # 224x224 모델 학습/평가
├── force_estimation_1632x1080.py    # 원본 해상도 모델 학습/평가
├── resolution_study.py              # 해상도별 성능 비교 실험
├── vid_process_RGB.py               # 단일 영상 전처리 (RGB)
├── vid_process_RG                   # 단일 영상 전처리 (RG)
├── vid_process_RGB_img_for_all.py   # 일괄 영상 전처리
├── FTdata_sync.py                   # 단일 센서 동기화
├── FTdata_sync_for_all.py           # 일괄 센서 동기화
└── .gitignore
```
