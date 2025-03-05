import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import os

# Función para redimensionar manteniendo la relación de aspecto
def resize_with_aspect_ratio(image, target_size):
    """
    Redimensiona la imagen manteniendo la relación de aspecto y la rellena con un color de fondo.
    :param image: Imagen original.
    :param target_size: Tamaño deseado (ancho, alto).
    :return: Imagen redimensionada y rellenada.
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calcular la relación de aspecto
    aspect_ratio = w / h
    target_aspect_ratio = target_w / target_h

    # Redimensionar manteniendo la relación de aspecto
    if aspect_ratio > target_aspect_ratio:
        # La imagen es más ancha que el objetivo
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:
        # La imagen es más alta que el objetivo
        new_h = target_h
        new_w = int(target_h * aspect_ratio)

    resized_image = cv2.resize(image, (new_w, new_h))

    # Crear una imagen de fondo con el tamaño deseado
    padded_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Calcular la posición para centrar la imagen redimensionada
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # Pegar la imagen redimensionada en el centro de la imagen de fondo
    padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

    return padded_image

# Cargar modelo
model = load_model("mi_modelo_señas_manos.h5")

# Etiquetas (deben coincidir con las carpetas del dataset)
dataset_path = "dataset_manos"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"No se encontró el dataset en la ruta: {dataset_path}")
CLASSES = sorted(os.listdir(dataset_path))  # ['A', 'B', 'C', ...]

# Iniciar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Configuración de la cámara
cap = cv2.VideoCapture(0)

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
                hand_img_resized = resize_with_aspect_ratio(hand_img, (230, 250))  # Cambia el tamaño según tu modelo
                
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
    
    cv2.imshow("Deteccion de señas", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
