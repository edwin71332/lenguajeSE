import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Función para redimensionar manteniendo la relación de aspecto
def resize_with_aspect_ratio(image, target_size):
    h, w = image.shape[:2]
    target_w, target_h = target_size
    aspect_ratio = w / h
    target_aspect_ratio = target_w / target_h

    if aspect_ratio > target_aspect_ratio:
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:
        new_h = target_h
        new_w = int(target_h * aspect_ratio)

    resized_image = cv2.resize(image, (new_w, new_h))
    padded_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

    return padded_image

# Cargar modelo
model = load_model("modelo_vocales.h5")

# Definir manualmente las clases 
CLASSES = ['A', 'E', 'I', 'O', 'U']

'''CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','K',   
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y'
] '''
 

# Iniciar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Configuración de la cámara
cap = cv2.VideoCapture(0)

# Tamaño deseado para la ventana de la cámara
window_width = 1280
window_height = 720

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertir a RGB y detectar manos
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Obtener coordenadas de la mano
            h, w, _ = frame.shape
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * h
            
            # Expandir el área de la mano un 20%
            expand = 0.2
            x_min = max(0, x_min - expand * (x_max - x_min))
            x_max = min(w, x_max + expand * (x_max - x_min))
            y_min = max(0, y_min - expand * (y_max - y_min))
            y_max = min(h, y_max + expand * (y_max - y_min))
            
            # Dibujar un cuadrado alrededor de la mano
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            
            # Recortar la mano
            hand_img = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
            
            if hand_img.size != 0:  # Verificar que la imagen no esté vacía
                # Redimensionar manteniendo la relación de aspecto
                hand_img_resized = resize_with_aspect_ratio(hand_img, (200, 200))
                
                # Preprocesar la imagen
                img = np.expand_dims(hand_img_resized, axis=0) / 255.0
                
                # Predecir
                prediction = model.predict(img)
                class_idx = np.argmax(prediction)
                class_name = CLASSES[class_idx]
                confidence = np.max(prediction)
                
                # Mostrar resultado
                cv2.putText(frame, f"{class_name} ({confidence:.2f})", (int(x_min), int(y_min) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Redimensionar el frame para aumentar el tamaño de la ventana
    frame_resized = cv2.resize(frame, (window_width, window_height))
    
    cv2.imshow("Deteccion de señas", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
