import tensorflow as tf
#importar modelo
model = tf.keras.models.load_model("modeloPractica_v6.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# por si se quieren el modelo mas ligero
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("modeloPractica_v6_float32.tflite", "wb") as f:
    f.write(tflite_model)

print("Conversión completada: modeloPractica_v6_float32.tflite")
