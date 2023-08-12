
import os, random

import common as com
import numpy as np
import tensorflow as tf

def representative_data_gen():
    dir = "/path/to/dev_data/" + machine_type + "/train/"
    filenames = random.sample(os.listdir(dir), 100)

    for audio_file in filenames:
        file_path = dir + audio_file

        try:
            data = com.file_to_vectors(file_path,
                                       n_mels=param["feature"]["n_mels"],
                                       n_frames=param["feature"]["n_frames"],
                                       n_fft=param["feature"]["n_fft"],
                                       hop_length=param["feature"]["hop_length"],
                                       power=param["feature"]["power"])
        except:
            com.logger.error("File broken: {}".format(file_path))

        # 1D vector to 2D image
        data = data.reshape(data.shape[0], param["feature"]["n_frames"], param["feature"]["n_mels"], 1)

        data = np.array(data, dtype=np.float32)
        yield [data]

param = com.yaml_load()

machine_types = ["gearbox", "fan", "pump", "slider", "ToyCar", "ToyTrain", "valve"]

for machine_type in machine_types:
    model_file = "models_keras/model_" + machine_type +  ".hdf5"
    model = tf.keras.models.load_model(model_file)

    # Convert Keras model to a tflite model with post-training integer quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Set the input tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8

    tflite_model = converter.convert()

    with open('models_compressed/model_' + machine_type + ".tflite", 'wb') as f:
        f.write(tflite_model)