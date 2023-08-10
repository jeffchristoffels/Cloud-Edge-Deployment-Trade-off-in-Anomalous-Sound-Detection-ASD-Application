
########################################################################
# import default libraries
########################################################################

import os
import sys
import gc
import time
import random
import argparse
import yaml

########################################################################

start_time = time.time()

########################################################################
# import additional libraries
########################################################################
import numpy as np
import tensorflow as tf                         # If hardware can run tensorflow
# import tflite_runtime.interpreter as tflite   # If hardware can only run tensorflow lite
import scipy.stats

try:
    from sklearn.externals import joblib
except:
    import joblib

import common as com

########################################################################

########################################################################
# load information
########################################################################

param = com.yaml_load()
with open("directories.yaml") as stream:
    directories = yaml.safe_load(stream)

# Parse arguments of command line
parser = argparse.ArgumentParser(description = 'Without option argument, it will not run.')
parser.add_argument("machine_type")
parser.add_argument("section_name")
parser.add_argument("optimization")
parser.add_argument("audio_amount")
args = parser.parse_args()

machine_type = args.machine_type
section_name = args.section_name
optimization = args.optimization
audio_amount = int(args.audio_amount)

########################################################################

print("Inference speed test for compressed MobileNet-v2 with " + optimization)
print("Time needed for imports: " + str(time.time() - start_time) + " s")

########################################################################

started_inference = False       # Don't take into account first inference time
inference_times = []
anomaly_scores = []
decision_results = []

start_time = time.time()

model_file = "{model_directory}/mobilenetv2/{optimization}/model_{machine_type}.tflite" \
    .format(model_directory = directories["model_directory"],
            optimization = optimization,
            machine_type = machine_type)

interpreter = tf.lite.Interpreter(model_path = model_file)
input_details = interpreter.get_input_details()
interpreter.resize_tensor_input(0, [250, 64, 128, 1])

# load section names for conditioning
section_names_file_path = "{model_directory}/mobilenetv2/{optimization}/section_names_{machine_type}.pkl" \
    .format(model_directory = directories["model_directory"],
            optimization = optimization,
            machine_type = machine_type)
trained_section_names = joblib.load(section_names_file_path)
n_sections = trained_section_names.shape[0]

# load anomaly score distribution for determining threshold
score_distr_file_path = "{model_directory}/mobilenetv2/{optimization}/score_distr_{machine_type}.pkl" \
    .format(model_directory = directories["model_directory"],
            optimization = optimization,
            machine_type = machine_type)
shape_hat, loc_hat, scale_hat = joblib.load(score_distr_file_path)

# determine threshold for decision
decision_threshold = scipy.stats.gamma.ppf(q=param["decision_threshold"], a=shape_hat, loc=loc_hat, scale=scale_hat)

print("Loaded data for " + machine_type + " in " + str(time.time() - start_time) + " s")

dir = "{audio_directory}/{machine_type}/target_test/" \
    .format(audio_directory=directories["audio_directory"],
            machine_type=machine_type)
all_files = os.listdir(dir)

temp_array = np.nonzero(trained_section_names == section_name)[0]
if temp_array.shape[0] == 0:
    print("Section not trained: exiting.")
    sys.exit()
else:
    section_idx = temp_array[0]

section_files = []
for file in all_files:
    if file.startswith(section_name):
        section_files.append(file)

filenames = random.sample(section_files, audio_amount + 1)

for audio_file in filenames:

    start_time = time.time()

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

    # make one-hot vector for conditioning
    condition = np.zeros((data.shape[0], n_sections), float)
    condition[:, section_idx: section_idx + 1] = 1

    # 1D vector to 2D image
    data = data.reshape(data.shape[0], param["feature"]["n_frames"], param["feature"]["n_mels"], 1)
    input_data = np.array(data, dtype=input_details[0]['dtype'])

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], input_data)

    interpreter.invoke()

    p = interpreter.get_tensor(output_details[0]['index'])[:, section_idx: section_idx + 1]
    y_pred = np.mean(np.log(np.maximum(1.0 - p, sys.float_info.epsilon)
                            - np.log(np.maximum(p, sys.float_info.epsilon))))

    # Save anomaly score
    anomaly_scores.append(y_pred)

    # Save decision results
    decision_threshold = scipy.stats.gamma.ppf(q=param["decision_threshold"], a=shape_hat, loc=loc_hat, scale=scale_hat)

    if y_pred > decision_threshold:
        decision_results.append("anomalous")
    else:
        decision_results.append("normal")

    if not started_inference:
        started_inference = True
        continue

    inference_times.append(time.time() - start_time)

del data
del interpreter
gc.collect()

mean = np.mean(inference_times)
standard_deviation = np.std(inference_times)

print("Mean inference time for " + section_name +  " of machine type '" + machine_type + "' is " + str(mean) + " s "
        "with standard deviation of " + str(standard_deviation))
