import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import glob

# --- 설정부 ---
FPS = 20.0  # 원본 영상의 FPS (시간 계산용)
MS_PER_FRAME = 1000.0 / FPS

def get_image_sync_points(image_files):
    """이미지 리스트를 넘기며 두 곳의 싱크 지점(index)을 추출합니다."""
    if not image_files:
        print("오류: 처리할 이미지 파일이 없습니다.")
        return None

    total_frames = len(image_files)
    curr_idx = 0
    idx_start, idx_end = None, None

    print(f"\n--- [이미지 싱크] 총 {total_frames} 프레임 ---")
    print("D:+1 | A:-1 | F:+10 | S:-10 | W:+100 | Q:-100")
    print("1: 시작 지점 설정 | 2: 끝 지점 설정 | Enter: 완료 | ESC: 건너뛰기")

    while True:
        # 이미지 읽기 (한글 경로 대응)
        img_array = np.fromfile(image_files[curr_idx], np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            break

        # 정보 표시
        t_ms = curr_idx * MS_PER_FRAME
        info = f"Frame: {curr_idx}/{total_frames-1} ({t_ms:.1f}ms)"
        s_info = f"Start(1): {f'{idx_start} ({(idx_start*MS_PER_FRAME):.1f}ms)' if idx_start is not None else 'Not Set'}"
        e_info = f"End(2): {f'{idx_end} ({(idx_end*MS_PER_FRAME):.1f}ms)' if idx_end is not None else 'Not Set'}"
        
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, s_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, e_info, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Image Sync Tool", frame)
        key = cv2.waitKey(0)

        if key == ord('d'): curr_idx = min(curr_idx + 1, total_frames - 1)
        elif key == ord('a'): curr_idx = max(curr_idx - 1, 0)
        elif key == ord('f'): curr_idx = min(curr_idx + 10, total_frames - 1)
        elif key == ord('s'): curr_idx = max(curr_idx - 10, 0)
        elif key == ord('w'): curr_idx = min(curr_idx + 100, total_frames - 1)
        elif key == ord('q'): curr_idx = max(curr_idx - 100, 0)
        elif key == ord('1'):
            idx_start = curr_idx
            print(f"시작 프레임(1) 설정: {idx_start}")
        elif key == ord('2'):
            idx_end = curr_idx
            print(f"끝 프레임(2) 설정: {idx_end}")
        elif key == 13: # Enter
            if idx_start is not None and idx_end is not None:
                break
            else:
                print("오류: 시작과 끝 지점을 모두 설정해야 합니다.")
        elif key == 27: # ESC
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    return idx_start, idx_end

def run_sync_for_image_folder(image_dir, sensor_csv_path, output_csv_path):
    """이미지 폴더와 센서 데이터를 동기화합니다."""
    # 1. 이미지 리스트 확보
    image_files = sorted(list(image_dir.glob("*.jpg")))
    if not image_files:
        return

    # 2. 센서 데이터 로드 (기존 로직 유지)
    if not sensor_csv_path.exists():
        print(f"오류: 센서 파일 없음: {sensor_csv_path}")
        return
    df_sensor = pd.read_csv(sensor_csv_path).iloc[6:].reset_index(drop=True)

    # 3. 수동 싱크 지점 획득
    sync_res = get_image_sync_points(image_files)
    if not sync_res: return
    idx_v_start, idx_v_end = sync_res
    
    # 시간(ms)으로 변환
    vt_start = idx_v_start * MS_PER_FRAME
    vt_end = idx_v_end * MS_PER_FRAME

    # 4. 센서 인덱스 입력 및 보간 (기존 로직 유지)
    print(f"\n--- [센서 인덱스 입력] {sensor_csv_path.name} ---")
    try:
        idx_start_input = int(input(f"영상 시작프레임({idx_v_start})에 대응하는 센서 행 번호: "))
        idx_end_input = int(input(f"영상 끝프레임({idx_v_end})에 대응하는 센서 행 번호: "))
        idx_start = idx_start_input - 8
        idx_end = idx_end_input - 8
    except:
        return

    actual_period = (vt_end - vt_start) / (idx_end - idx_start)
    df_sensor['synced_time (ms)'] = (df_sensor.index - idx_start) * actual_period + vt_start
    df_sensor['sensor_original_index'] = df_sensor.index + 7

    # 5. 프레임별 매칭
    matched_data = []
    sensor_times = df_sensor['synced_time (ms)'].values
    
    for i in range(len(image_files)):
        v_t = i * MS_PER_FRAME
        nearest_idx = (np.abs(sensor_times - v_t)).argmin()
        matched_row = df_sensor.iloc[nearest_idx].copy()
        matched_row['video_time (ms)'] = v_t
        matched_row['image_path'] = str(image_files[i].relative_to(image_dir.parent.parent))
        matched_data.append(matched_row)

    df_final = pd.DataFrame(matched_data).reset_index(drop=True)
    df_final.to_csv(output_csv_path, index=False)
    print(f"✓ 저장 완료: {output_csv_path.name}")

def main():
    # 경로 설정 (사용자 환경에 맞춰 자동 구성)
    data_root = Path(r"C:\Users\hq\Documents\20251202_Force_Estimation\data")

    subfolders = [f"{i:02d}" for i in range(41, 51)]
    
    for folder_name in subfolders:
        current_folder = data_root / folder_name
        if not current_folder.exists(): continue
        
        for v_idx in range(1, 10):
            image_dir = current_folder / str(v_idx)  # 예: data/01/1/
            sensor_csv_path = current_folder / f"{v_idx}.csv" # 예: data/01/1.csv
            output_csv_path = current_folder / f"{v_idx}_frame_synced.csv"

            if image_dir.exists() and sensor_csv_path.exists():
                if output_csv_path.exists():
                    print(f"이미 존재함: {output_csv_path.name}")
                    continue
                run_sync_for_image_folder(image_dir, sensor_csv_path, output_csv_path)

if __name__ == "__main__":
    main()