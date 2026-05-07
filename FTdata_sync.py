import pandas as pd
import numpy as np
import cv2
from pathlib import Path

def process_sensor_data(base_dir, base_name):
    """기본 전처리: 버퍼 삭제 및 데이터 로드"""
    csv_path = Path(base_dir) / f"{base_name}.csv"
    if not csv_path.exists():
        print(f"파일을 찾을 수 없습니다: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    # 첫 6개 데이터(buffer) 삭제 및 인덱스 초기화
    df_processed = df.iloc[6:].reset_index(drop=True)
    return df_processed

def get_video_sync_points(video_path):
    """영상을 재생하며 두 곳의 싱크 지점(1: 시작, 2: 끝)을 추출합니다."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    curr_frame = 0
    t_start, t_end = None, None

    print("\n[1단계: 영상 싱크 지점 선택]")
    print("D:+1 | A:-1 | F:+10 | S:-10")
    print("1: 시작 지점 설정 | 2: 끝 지점 설정 | Enter: 완료 (1, 2 모두 설정 후)")

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        ret, frame = cap.read()
        if not ret: break

        # 화면 정보 표시
        info = f"Frame: {curr_frame}/{total_frames-1}"
        s_info = f"Start(1): {f'{t_start:.2f}ms' if t_start is not None else 'Not Set'}"
        e_info = f"End(2): {f'{t_end:.2f}ms' if t_end is not None else 'Not Set'}"
        
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, s_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, e_info, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Video Sync Tool", frame)
        key = cv2.waitKey(0)

        if key == ord('d'): curr_frame = min(curr_frame + 1, total_frames - 1)
        elif key == ord('a'): curr_frame = max(curr_frame - 1, 0)
        elif key == ord('f'): curr_frame = min(curr_frame + 10, total_frames - 1)
        elif key == ord('s'): curr_frame = max(curr_frame - 10, 0)
        elif key == ord('1'):
            t_start = cap.get(cv2.CAP_PROP_POS_MSEC)
            print(f"시작 지점(1) 설정: {t_start:.2f} ms")
        elif key == ord('2'):
            t_end = cap.get(cv2.CAP_PROP_POS_MSEC)
            print(f"끝 지점(2) 설정: {t_end:.2f} ms")
        elif key == 13: # Enter
            if t_start is not None and t_end is not None:
                break
            else:
                print("시작 지점(1)과 끝 지점(2)을 모두 설정해야 합니다.")
        elif key == 27: # ESC
            return None

    cap.release()
    cv2.destroyAllWindows()
    return t_start, t_end

# --- 실행부 ---
base_name = "rnd3"
base_dir = Path(r"C:\Users\hq\OneDrive - 대전동신과학고등학교\KAIST_BS\2025\2025 URP\VBTS 제작\20251202_Force_Estimation\dataset1_hexagon")
video_path = base_dir / "rnd3_color_diff.mp4"

df_sensor = process_sensor_data(base_dir, base_name)

if df_sensor is not None:
    vt_syncs = get_video_sync_points(video_path)
    
    if vt_syncs:
        vt_start, vt_end = vt_syncs
        
        # 2단계: CSV 인덱스 입력
        print(f"\n[2단계: CSV 싱크 인덱스 입력]")
        idx_start = int(input(f"영상 시작점({vt_start:.2f}ms)에 대응하는 센서 data 엑셀 index: ")) - 8 # 버퍼 6개 + 2개 오프셋
        idx_end = int(input(f"영상 끝점({vt_end:.2f}ms)에 대응하는 센서 data 엑셀 index: ")) - 8
        
        # 3단계: 실제 주기 계산 및 전체 시간 할당
        actual_period = (vt_end - vt_start) / (idx_end - idx_start)
        df_sensor['synced_time (ms)'] = (df_sensor.index - idx_start) * actual_period + vt_start
        df_sensor['sensor_original_index'] = df_sensor.index # 원본 인덱스 보존

        # 4단계: 비디오의 모든 프레임 타임스탬프 추출
        print("\n[3단계: 비디오 타임스탬프 추출 및 매칭 중...]")
        cap = cv2.VideoCapture(str(video_path))
        video_timestamps = []
        while cap.isOpened():
            ret = cap.grab()
            if not ret: break
            video_timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        cap.release()

        # 5단계: 각 비디오 프레임 시각에 가장 가까운 센서 데이터 찾기
        matched_data = []
        sensor_times = df_sensor['synced_time (ms)'].values
        
        for v_t in video_timestamps:
            # 비디오 시각(v_t)과 센서 보간 시각 사이의 절대 차이가 최소인 지점
            nearest_idx = (np.abs(sensor_times - v_t)).argmin()
            matched_row = df_sensor.iloc[nearest_idx].copy()
            matched_row['video_time (ms)'] = v_t
            matched_data.append(matched_row)

        # 6단계: 데이터프레임 재구성 및 컬럼 순서 조정
        df_final = pd.DataFrame(matched_data).reset_index(drop=True)
        
        # 컬럼 순서 정의: [비디오 타임스탬프, 매칭된 센서 시각, 센서 인덱스, 나머지 데이터...]
        cols = ['video_time (ms)', 'synced_time (ms)', 'sensor_original_index']
        other_cols = [c for c in df_final.columns if c not in cols]
        df_final = df_final[cols + other_cols]

        # 7단계: 최종 CSV 저장
        final_output_path = base_dir / f"{base_name}_frame_synced.csv"
        df_final.to_csv(final_output_path, index=False)
        
        print(f"\n매칭 완료! (총 {len(df_final)} 프레임)")
        print(f"최종 파일: {final_output_path}")