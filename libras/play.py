"""
Passo 4 - Reconhecimento de sinais de Libras pela webcam.
Eduardo Sochodolak e Andrew Bertelli

Uso:  python play.py
  q = sair
"""
import json
from collections import Counter, deque

import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 224
model = tf.keras.models.load_model("gesture_model.keras")
class_names = json.load(open("class_names.json"))

def predict(roi):
    img = cv2.cvtColor(cv2.resize(roi, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    probs = model.predict(img[None].astype("float32"), verbose=0)[0]
    return class_names[int(np.argmax(probs))], float(np.max(probs)), probs

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Nao consegui abrir a webcam.")

recent = deque(maxlen=7)           # suaviza: voto da maioria nos ultimos frames

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    side = min(h, w) // 2 + 60
    x1, y1 = w // 2 - side // 2, h // 2 - side // 2
    roi = frame[y1:y1 + side, x1:x1 + side]

    label, conf, probs = predict(roi)
    recent.append(label)
    stable = Counter(recent).most_common(1)[0][0]
    sinal = "Mostre um sinal no quadrado" if stable == "nada" else f"Sinal: {stable}"

    cv2.rectangle(frame, (x1, y1), (x1 + side, y1 + side), (0, 255, 0), 3)
    cv2.putText(frame, f"Modelo ve: {label} ({conf:.0%})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    # barras de probabilidade (otimo para mostrar a "incerteza" do modelo)
    for i, c in enumerate(class_names):
        bw = int(probs[i] * 150)
        cv2.rectangle(frame, (10, 50 + i * 22), (10 + bw, 64 + i * 22), (0, 200, 255), -1)
        cv2.putText(frame, c, (170, 63 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, sinal, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("Sinais de Libras com CNN", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
