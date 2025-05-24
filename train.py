from ultralytics import YOLO
import cv2
import numpy as np

# === PARAMETRI ===
MODEL_PATH = "runs/detect/train2/weights/best.pt"  # il tuo modello addestrato
CLASSES_PATH = "Riconoscimento-Biometria-Facciale/Multi Face Biometry.v1i.yolov11/data.yaml"

NOISE_SIGMA = 25
OPACITY_ALPHA = 1.0

# === FUNZIONI PER EFFETTI VISIVI ===
def add_gaussian_noise(image, mean=0, sigma=25):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_opacity(image, alpha=0.7):
    overlay = image.copy()
    output = image.copy()
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output

# === CARICAMENTO MODELLO ===
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

print("Face recognition avviato. Premi ESC per uscire.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Applica effetti visivi
    noisy_frame = add_gaussian_noise(frame, sigma=NOISE_SIGMA)
    processed_frame = add_opacity(noisy_frame, alpha=OPACITY_ALPHA)
    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

    # Rilevamento
    results = model(frame_rgb, verbose=False)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]  # nome persona, es: 'marco', 'simone', ecc.

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(processed_frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    # Mostra il risultato
    cv2.imshow("Face Recognition", processed_frame)
    
    # Premi ESC per uscire
    if cv2.waitKey(1) == 27:
        print("Face recognition terminato.")
        break

# Rilascia la webcam e chiudi le finestre
cap.release()
cv2.destroyAllWindows()