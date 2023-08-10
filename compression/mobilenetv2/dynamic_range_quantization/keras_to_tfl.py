
import tensorflow as tf

machine_types = ["gearbox", "fan", "pump", "slider", "ToyCar", "ToyTrain", "valve"]

for machine_type in machine_types:
    model_file = "models_keras/model_" + machine_type +  ".hdf5"
    model = tf.keras.models.load_model(model_file)

    # Convert Keras model to a tflite model with post-training dynamic range quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open('models_compressed/model_' + machine_type + ".tflite", 'wb') as f:
        f.write(tflite_model)
