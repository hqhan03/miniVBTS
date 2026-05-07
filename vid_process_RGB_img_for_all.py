import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os

# --- 1. 설정부 ---
CROP_PERCENT_TOP = 0.0
CROP_PERCENT_BOTTOM = 0.0
CROP_PERCENT_LEFT = 0.0
CROP_PERCENT_RIGHT = 0.0

GREEN_CONTRAST_FACTOR = 20.0
RED_CONTRAST_FACTOR = 10.0
GAUSSIAN_KERNEL_SIZE = (5, 5)

# 경로 설정 
data_root = Path(r"C:\Users\hq\Documents\20251202_Force_Estimation\data")

def process_single_video(args):
    """
    별도의 프로세스에서 실행될 함수
    args: (video_path, save_dir)
    """
    video_path, save_dir = args
    save_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return f"Error: {video_path.name}"

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 크롭 설정
    crop_y1, crop_y2 = int(frame_height * CROP_PERCENT_TOP), int(frame_height * (1.0 - CROP_PERCENT_BOTTOM))
    crop_x1, crop_x2 = int(frame_width * CROP_PERCENT_LEFT), int(frame_width * (1.0 - CROP_PERCENT_RIGHT))
    is_cropping = (crop_y1 < crop_y2) and (crop_x1 < crop_x2)

    # --- 기준 프레임 계산 ---
    num_avg = 5
    accumulator = np.zeros((frame_height, frame_width, 3), dtype=np.float64)
    actual_avg_count = 0
    for _ in range(num_avg):
        ret, frame = cap.read()
        if not ret: break
        accumulator += frame.astype(np.float64)
        actual_avg_count += 1

    if actual_avg_count == 0:
        cap.release()
        return f"Empty: {video_path.name}"

    reference_gray = cv2.cvtColor((accumulator / actual_avg_count).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.GaussianBlur(reference_gray, GAUSSIAN_KERNEL_SIZE, 0).astype(np.int16)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- 메인 처리 루프 ---
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, GAUSSIAN_KERNEL_SIZE, 0).astype(np.int16)

        # 차이 계산 및 채널 생성 (벡터화 연산 최적화)
        diff = frame_gray - reference_gray
        
        red_channel = np.clip(np.maximum(0, -diff) * RED_CONTRAST_FACTOR, 0, 255).astype(np.uint8)
        green_channel = np.clip(np.maximum(0, diff) * GREEN_CONTRAST_FACTOR, 0, 255).astype(np.uint8)
        blue_channel = reference_gray.astype(np.uint8) # 원본 reference 활용

        color_diff_frame = cv2.merge([blue_channel, green_channel, red_channel])
        
        if is_cropping:
            color_diff_frame = color_diff_frame[crop_y1:crop_y2, crop_x1:crop_x2]

        # 저장 (한글 경로 대응)
        frame_filename = save_dir / f"{i:06d}.jpg"
        result, encoded_img = cv2.imencode('.jpg', color_diff_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if result:
            with open(frame_filename, mode='wb') as f:
                f.write(encoded_img)

    cap.release()
    return f"Done: {video_path.name}"

# --- 2. 실행부 ---
if __name__ == "__main__":
    if not data_root.exists():
        print(f"데이터 경로 없음: {data_root}")
    else:
        # 작업 리스트 생성
        task_list = []
        subfolders = [f"{i:02d}" for i in range(26, 51)]
        
        for folder_name in subfolders:
            current_folder = data_root / folder_name
            if not current_folder.exists(): continue
            
            for v_idx in range(1, 10):
                video_file = current_folder / f"{v_idx}.mp4"
                if video_file.exists():
                    save_path = current_folder / video_file.stem
                    task_list.append((video_file, save_path))

        # 멀티프로세싱 실행 (CPU 코어 수에 맞춰 자동 설정)
        print(f"총 {len(task_list)}개의 비디오를 병렬 처리합니다...")
        # max_workers=os.cpu_count() // 2 정도로 조절 가능 (메모리 부족 시)
        with ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(process_single_video, task_list), total=len(task_list), desc="Overall Progress"))

        print("\n모든 영상 처리가 완료되었습니다.")