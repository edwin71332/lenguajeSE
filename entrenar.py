import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configuración
DIR_DATASET = "dataset_manos"    # Carpeta con las imágenes de las manos
IMG_SIZE = (200, 200)            # Tamaño de las imágenes
BATCH_SIZE = 32
EPOCHS = 100                      

# Generador de datos con mayor aumento de datos
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Normalización específica de MobileNetV2
    rotation_range=30,                          # Aumenta el rango de rotación
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],                # Variación en brillo
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2                        # 20% de datos para validación
)

# Cargar datos para entrenamiento y validación
train_generator = train_datagen.flow_from_directory(
    DIR_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DIR_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# Transfer Learning con MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False  # Congelar la base para no sobreajustar con pocos datos

# Construcción del modelo final
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Reduce las dimensiones manteniendo la información global
    layers.BatchNormalization(),
    layers.Dropout(0.5),                # Aumenta la regularización
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks para evitar sobreajuste y guardar el mejor modelo
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint("mi_modelo_señas_manos1.h5", monitor='val_loss', 
                             save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

callbacks = [early_stop, checkpoint, reduce_lr]

# Entrenar el modelo
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

print("Entrenamiento finalizado y modelo guardado (se conserva el de menor val_loss).")


import subprocess

 
# Después de finalizar el entrenamiento:
subprocess.run(["python", "entrenar2.py"])
