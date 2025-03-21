import cv2
import mediapipe as mp
import numpy as np
import os

label = "hello"
save_dir = f"data/{label}"
os.makedirs(save_dir, exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

count = 0

while count < 200:
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
            np.save(f"{save_dir}/{label}_{count}.npy", np.array(landmark_list))
            print(f"Saved: {label}_{count}")
            count += 1

    cv2.imshow('Collecting...', frame)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
