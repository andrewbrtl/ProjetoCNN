"""
Passo 4 - Coleta dos sinais de Libras pela webcam.
Eduardo Sochodolak e Andrew Bertelli

Uso:  python collect.py          (para em 300 fotos por sinal)
      python collect.py 240      (para em 240, usado na coleta em etapas)

IMPORTANTE: clique uma vez na JANELA DO VIDEO para ela receber as teclas
(se o terminal estiver em foco, nada acontece).

Teclas (aperte UMA vez, nao segure):
  1 ou a = A    2 ou b = B    3 ou c = C    4 ou d = D    5 ou i = I
  6 ou l = L    7 ou o = O    8 ou v = V    9 ou w = W    0 ou y = Y
  n = nada
  Ao apertar, ha uma contagem de 3s para voce posicionar a mao, e a
  gravacao comeca sozinha. Aperte a mesma tecla de novo para pausar.
  q = sair
"""
import os
import sys
import time
import cv2

KEYS = {
    ord("1"): "A", ord("a"): "A",
    ord("2"): "B", ord("b"): "B",
    ord("3"): "C", ord("c"): "C",
    ord("4"): "D", ord("d"): "D",
    ord("5"): "I", ord("i"): "I",
    ord("6"): "L", ord("l"): "L",
    ord("7"): "O", ord("o"): "O",
    ord("8"): "V", ord("v"): "V",
    ord("9"): "W", ord("w"): "W",
    ord("0"): "Y", ord("y"): "Y",
    ord("n"): "nada",
}
CLASSES = ["A", "B", "C", "D", "I", "L", "O", "V", "W", "Y", "nada"]
DATA_DIR = "data"
IMG_SIZE = 224
SAVE_EVERY_N_FRAMES = 3
MAX_PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
COUNTDOWN = 3.0
DEBOUNCE = 0.5   # ignora repeticao de tecla (segurar) por 0.5s

teclas = {c: "/".join(chr(k) for k, v in KEYS.items() if v == c) for c in CLASSES}

for c in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, c), exist_ok=True)

def count(cls):
    return len([f for f in os.listdir(os.path.join(DATA_DIR, cls)) if f.endswith(".jpg")])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Nao consegui abrir a webcam. Feche outros apps que usam a camera "
                     "(Zoom, Meet...) ou tente cv2.VideoCapture(1).")

recording = None       # classe sendo gravada
countdown_start = None
last_key_time = 0
frame_count = 0
WIN = "Coleta"
print("Janela aberta. Clique nela e aperte 1-9/0 (ou a letra do sinal), n = nada. 'q' sai.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Falha ao ler a webcam.")
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    side = min(h, w) // 2 + 60
    x1, y1 = w // 2 - side // 2, h // 2 - side // 2
    roi = frame[y1:y1 + side, x1:x1 + side]

    status, color = "PARADO - aperte a tecla do sinal na janela", (0, 255, 0)
    if recording and countdown_start is not None:
        left = COUNTDOWN - (time.time() - countdown_start)
        if left > 0:
            status, color = f"{recording}: prepare a mao... {int(left) + 1}", (0, 255, 255)
        else:
            countdown_start = None
    if recording and countdown_start is None:
        status, color = f"GRAVANDO: {recording}", (0, 0, 255)
        frame_count += 1
        if frame_count % SAVE_EVERY_N_FRAMES == 0:
            if count(recording) < MAX_PER_CLASS:
                img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(os.path.join(DATA_DIR, recording, f"{int(time.time() * 1000)}.jpg"), img)
            else:
                print(f"{recording}: completo ({MAX_PER_CLASS} fotos).")
                recording = None

    cv2.rectangle(frame, (x1, y1), (x1 + side, y1 + side), color, 3)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    for i, c in enumerate(CLASSES):
        cv2.putText(frame, f"[{teclas[c]}] {c}: {count(c)}/{MAX_PER_CLASS}", (10, 65 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow(WIN, frame)

    key = cv2.waitKey(10) & 0xFF
    if key == 255:
        continue
    if key == ord("q"):
        break
    if key in KEYS and time.time() - last_key_time > DEBOUNCE:
        last_key_time = time.time()
        cls = KEYS[key]
        if recording == cls:
            recording = None
            print(f"Pausado: {cls}")
        else:
            recording, countdown_start = cls, time.time()
            print(f"Preparando: {cls}")
    elif key not in KEYS:
        print(f"(tecla nao reconhecida: codigo {key})")

cap.release()
cv2.destroyAllWindows()
print({c: count(c) for c in CLASSES})
