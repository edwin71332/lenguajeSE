import cv2
import mediapipe as mp
import os
import time
import numpy as np
import string

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

# Configuración
LETRAS = list(string.ascii_uppercase)  # Letras que quieres entrenar
IMAGENES_POR_LETRA = 100   # Número de imágenes por letra
DIR_DATASET = "dataset_manos"
IMG_SIZE = (200, 200)      # Tamaño deseado para las imágenes

# Crear directorio principal
if not os.path.exists(DIR_DATASET):
    os.makedirs(DIR_DATASET)

# Iniciar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Iniciar cámara
cap = cv2.VideoCapture(0)

# Variable para controlar la pausa
pausa = False

for letra in LETRAS:
    letra_dir = os.path.join(DIR_DATASET, letra)
    if not os.path.exists(letra_dir):
        os.makedirs(letra_dir)
    
    print(f"Presiona 's' para empezar a capturar la letra {letra}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir a RGB y detectar manos
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Dibujar un cuadrado alrededor de la mano
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
        
        cv2.putText(frame, f"Letra: {letra}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Captura de imagenes", frame)
        
        # Esperar a que se presione 's' para empezar
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    
    contador = 0
    while contador < IMAGENES_POR_LETRA:
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
                
                # Redimensionar manteniendo la relación de aspecto
                hand_img_resized = resize_with_aspect_ratio(hand_img, IMG_SIZE)
                
                # Guardar la imagen de la mano
                if hand_img_resized.size != 0:  # Verificar que la imagen no esté vacía
                    img_path = os.path.join(letra_dir, f"{letra}_{contador}.jpg")
                    cv2.imwrite(img_path, hand_img_resized)
                    print(f"Imagen guardada: {img_path}")
                    contador += 1
                    time.sleep(0.1)  # Espera para evitar imágenes duplicadas
        
        # Mostrar la imagen en la ventana
        cv2.imshow("Captura de imagenes", frame)
        
        # Controlar la pausa con la tecla 'p'
        key = cv2.waitKey(1)
        if key & 0xFF == ord('p'):
            pausa = not pausa  # Alternar entre pausa y reanudar
            print("Pausa activada" if pausa else "Pausa desactivada")
        
        # Si está en pausa, no avanzar
        while pausa:
            key = cv2.waitKey(1)
            if key & 0xFF == ord('p'):
                pausa = not pausa  # Reanudar
                print("Pausa desactivada")
                break

cap.release()
cv2.destroyAllWindows()