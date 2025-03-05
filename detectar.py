import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuración
DIR_DATASET = "dataset_manos"  # Carpeta con las imágenes de la mano
IMG_SIZE = (200, 200)          # Tamaño de las imágenes
BATCH_SIZE = 32
EPOCHS = 30

# Generador de datos con aumento
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Cargar datos
train_generator = train_datagen.flow_from_directory(
    DIR_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    DIR_DATASET,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# agg al modelo CNN regularización L2 y Dropout para evitar el sobreajuste
model = models.Sequential([
    # Primer bloque convolucional
    layers.Conv2D(32, (3, 3), activation='relu',
                  kernel_regularizer=regularizers.l2(0.001),
                  input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # Dropout para reducir el sobreajuste
    
    # Segundo bloque convolucional
    layers.Conv2D(64, (3, 3), activation='relu',
                  kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Tercer bloque convolucional
    layers.Conv2D(128, (3, 3), activation='relu',
                  kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Aplanamiento y capa densa
    layers.Flatten(),
    layers.Dense(256, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),  # Dropout mayor en la capa densa
    
    # Capa de salida
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# Compilación del modelo con una tasa de aprendizaje reducida para mayor estabilidad
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# Guardar el modelo entrenado
model.save("mi_modelo_señas_manos.h5")
print("Modelo guardado!")
