import cv2
import mediapipe as mp
import numpy as np
import os
import time

# 수집할 데이터 이름
label = "thankyou"
save_dir = f"data/{label}"
os.makedirs(save_dir, exist_ok=True)

# MediaPipe 손 인식 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# 웹캠 열기
cap = cv2.VideoCapture(0)

count = 0   # 저장할 프레임 수
prev_landmark = None    # 이전 프레임의 손 좌표
stable_frame = 0    # 연속으로 손이(거의) 안움직인 프레임 수 = 정지 상태
required_stable_frames = 20    # 정지 상태 간주 기준(약 2초)
collecting = False  # 현재 수집 중인지 여부

while cap.isOpened():   # 카메라가 열려있는 동안
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    landmark_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 시각화
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            # 좌표 저장
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

        # 양손 모두 인식된 경우에만 저장 (21 * 2 손가락 개수)
        if len(landmark_list) == 63 * 2:
            # 이전 프레임과 차이 계산
            if prev_landmark is not None:
                diff = np.linalg.norm(np.array(landmark_list) - np.array(prev_landmark))
                # 손이 거의 안움직이면 stable_frame 증가
                if diff < 0.05:
                    stable_frame += 1
                # 움직임 발생시 초기화
                else:
                    stable_frame = 0
            prev_landmark = landmark_list.copy()

            # 정지 상태 2초 유지 시 수집 시작
            if not collecting and stable_frame >= required_stable_frames:
                collecting = True

            # 수집 중이면 landmark 저장
            if collecting:
                np.save(f"{save_dir}/{label}_{count}.npy", np.array(landmark_list))
                print(f"Saved: {label}_{count}")
                count += 1

    # 현재 상태에 따라 화면에 표시
    if not collecting:
        cv2.putText(frame, f"정지 상태 대기 중... ({stable_frame}/{required_stable_frames})", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"{label} 수집 중: {count}/200", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
    # 화면에 출력   
    cv2.imshow('Collecting...', frame)

    # ESC 키를 누르거나 200개 다 수집하면 종료
    if cv2.waitKey(10) & 0xFF == 27 or count >= 200:
        break

cap.release()
cv2.destroyAllWindows()
