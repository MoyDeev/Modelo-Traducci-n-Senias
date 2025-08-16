import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import json
import os

TAMANO_IMG = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.0001

train_dir = "dataset/train"
val_dir = "dataset/val"
#data augmentation, sirve para agregar mas variedad en las imagenes
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    channel_shift_range=30.0,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(TAMANO_IMG, TAMANO_IMG),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(TAMANO_IMG, TAMANO_IMG),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# mobilenetv2
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(TAMANO_IMG, TAMANO_IMG, 3)
)

base_model.trainable = False  # congelar capas base al inicio

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
predictions = Dense(
    train_generator.num_classes,
    activation="softmax"
)(x)

model = Model(inputs=base_model.input, outputs=predictions)

# COMPILACIÓN
optimizer = Adam(learning_rate=LR)
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=["accuracy"]
)

# ENTRENAMIENTO
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1
)

# DESCONGELAR CAPAS (Fine-tuning)
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LR/10),
    loss=loss_fn,
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    verbose=1
)

# GUARDAR MODELO Y CLASES
model.save("modeloPractica_v6.h5")

with open("modeloPractica_v6.json", "w") as f:
    json.dump(train_generator.class_indices, f)

print("Modelo y clases guardados correctamente.")

# Graficas
# def plot_history(histories, title="Entrenamiento del modelo"):
#     plt.figure(figsize=(14, 5))

#     # Accuracy
#     plt.subplot(1, 2, 1)
#     for name, history in histories:
#         plt.plot(history.history['accuracy'], label=f"{name} acc")
#         plt.plot(history.history['val_accuracy'], label=f"{name} val_acc")
#     plt.title("Precisión")
#     plt.xlabel("Épocas")
#     plt.ylabel("Accuracy")
#     plt.legend()

#     # Loss
#     plt.subplot(1, 2, 2)
#     for name, history in histories:
#         plt.plot(history.history['loss'], label=f"{name} loss")
#         plt.plot(history.history['val_loss'], label=f"{name} val_loss")
#     plt.title("Pérdida")
#     plt.xlabel("Épocas")
#     plt.ylabel("Loss")
#     plt.legend()

#     plt.suptitle(title)
#     plt.show()

# plot_history([("Entrenamiento", history), ("Fine-tuning", history_finetune)])