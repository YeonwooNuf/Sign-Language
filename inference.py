import cv2
import mediapipe as mp
import numpy as np
import joblib

# 모델 불러오기
model = joblib.load("models/knn_model.pkl")

# Mediapipe 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    landmark_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

        # 양손 기준 126차원 입력이면 예측 시도(왼손 63/오른손 63)
        if len(landmark_list) == 126:
            input_data = np.array(landmark_list).reshape(1, -1)
            prediction = model.predict(input_data)[0]
            cv2.putText(frame, f"예측: {prediction}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        else:
            cv2.putText(frame, "손을 정확히 보여주세요", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

    else:
        cv2.putText(frame, "손이 감지되지 않음", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

    cv2.imshow("Sign Language Inference", frame)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
