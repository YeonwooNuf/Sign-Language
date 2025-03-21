import cv2
import mediapipe as mp
import numpy as np
import joblib
from PIL import ImageFont, ImageDraw, Image

# 모델 불러오기
model = joblib.load("models/knn_model.pkl")

# 영어 → 한글 매핑
label_map = {
    "hello": "안녕하세요",
    "thankyou": "감사합니다",
    "iloveyou": "사랑해요",
    "yes": "예",
    "no": "아니요"
}

# 한글 출력 함수 (D2Coding)
def draw_koreanText(frame, text, position=(30, 100), font_path="/Library/Fonts/D2Coding.ttf", font_size=40, color=(0, 0, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

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
            koreanText = label_map.get(prediction, "알 수 없음")    # 예측값 한글로 변환
            frame = draw_koreanText(frame, f"예측: {koreanText}")
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
