import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("model.h5")

def rep_data():
    for _ in range(100):
        yield [np.random.rand(1,5,1).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("model.tflite","wb") as f:
    f.write(tflite_model)

print("TFLite model ready")