# utils/preprocess.py

import os
import numpy as np
import pandas as pd

DATA_DIR = "data"
CSV_PATH = "dataset/all_landmarks.csv"

data = []
labels = []

for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    for file in os.listdir(label_dir):
        if file.endswith('.npy'):
            file_path = os.path.join(label_dir, file)
            landmark = np.load(file_path)

            # 양손 기준: 126개의 좌표 (21포인트 * 3D * 2손)
            if landmark.shape[0] == 126:
                data.append(landmark)
                labels.append(label)
            else:
                print(f"⚠️ 스킵됨: {file_path} → shape: {landmark.shape}")

# numpy 배열로 변환
X = np.array(data)
y = np.array(labels).reshape(-1, 1)

# 데이터 + 라벨 합치기
dataset = np.hstack([X, y])
columns = [f"x{i}" for i in range(X.shape[1])] + ["label"]
df = pd.DataFrame(dataset, columns=columns)

# 저장
os.makedirs("dataset", exist_ok=True)
df.to_csv(CSV_PATH, index=False)
print(f"✅ CSV 저장 완료: {CSV_PATH}")
print(f"샘플 수: {len(df)}개")
