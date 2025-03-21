import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib
import os

# 데이터 불러오기
df = pd.read_csv("dataset/all_landmarks.csv")

# 입력값(X), 라벨(y) 분리
X = df.drop("label", axis=1).values.astype(float)
y = df["label"].values

# 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 평가
y_pred = model.predict(X_test)
print("✅ 분류 결과:\n")
print(classification_report(y_test, y_pred))

# 모델 저장
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/knn_model.pkl")
print("✅ 모델 저장 완료: models/knn_model.pkl")
