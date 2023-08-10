
########################################################################
# import default libraries
########################################################################

import os
import sys
import gc
import time
import random
import yaml
import argparse

########################################################################

start_time = time.time()

########################################################################
# import additional libraries
########################################################################
import numpy as np
import tensorflow as tf
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
parser.add_argument("audio_amount")
args = parser.parse_args()

audio_amount = int(args.audio_amount)

########################################################################

print("Inference speed test for baseline MobileNet-v2")
print("Time needed for imports: " + str(time.time() - start_time) + " s")

machine_types = ["fan", "gearbox", "pump", "slider", "valve", "ToyCar", "ToyTrain"]
started_inference = False       # Don't take into account first inference time
inference_times = []
anomaly_scores = []
decision_results = []

for machine_type in machine_types:      # Test all machine types

    start_time = time.time()

    model_file = "{model_directory}/mobilenetv2/baseline/model_{machine_type}.hdf5" \
        .format(model_directory = directories["model_directory"],
                machine_type = machine_type)
    model =  tf.keras.models.load_model(model_file)

    # load section names for conditioning
    section_names_file_path = "{model_directory}/mobilenetv2/baseline/section_names_{machine_type}.pkl" \
        .format(model_directory = directories["model_directory"],
                machine_type = machine_type)
    trained_section_names = joblib.load(section_names_file_path)
    n_sections = trained_section_names.shape[0]

    # load anomaly score distribution for determining threshold
    score_distr_file_path = "{model_directory}/mobilenetv2/baseline/score_distr_{machine_type}.pkl" \
        .format(model_directory = directories["model_directory"],
                machine_type = machine_type)
    shape_hat, loc_hat, scale_hat = joblib.load(score_distr_file_path)

    # determine threshold for decision
    decision_threshold = scipy.stats.gamma.ppf(q=param["decision_threshold"], a=shape_hat, loc=loc_hat, scale=scale_hat)

    print("Loaded data for " + machine_type + " in " + str(time.time() - start_time) + " s")

    current_inference_times = []
    current_anomaly_scores = []
    current_decision_results = []

    dir = "{audio_directory}/{machine_type}/target_test/" \
        .format(audio_directory=directories["audio_directory"],
                machine_type=machine_type)
    all_files = os.listdir(dir)

    for section_name in trained_section_names:

        temp_array = np.nonzero(trained_section_names == section_name)[0]

        section_files = []
        for file in all_files:
            if file.startswith(section_name):
                section_files.append(file)

        if started_inference:
            filenames = random.sample(section_files, audio_amount + 1)
        else:
            filenames = random.sample(section_files, audio_amount)

        section_inference_times = []

        for audio_file in filenames:

            start_time = time.time()

            section_idx = temp_array[0]

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

            p = model.predict(data)[:, section_idx: section_idx + 1]
            y_pred = np.mean(np.log(np.maximum(1.0 - p, sys.float_info.epsilon)
                                    - np.log(np.maximum(p, sys.float_info.epsilon))))

            # Save anomaly score
            current_anomaly_scores.append(y_pred)

            # Save decision results
            decision_threshold = scipy.stats.gamma.ppf(q=param["decision_threshold"], a=shape_hat, loc=loc_hat, scale=scale_hat)

            if y_pred > decision_threshold:
                current_decision_results.append("anomalous")
            else:
                current_decision_results.append("normal")

            if not started_inference:
                started_inference = True
                continue

            section_inference_times.append(time.time() - start_time)

        current_inference_times.append(section_inference_times)

    del data
    del model
    gc.collect()

    inference_times.append(current_inference_times)
    anomaly_scores.append(current_anomaly_scores)
    decision_results.append(current_decision_results)

mean_inference_times = []
for i in range(len(inference_times)):
    for s in range(len(trained_section_names)):
        mean = np.mean(inference_times[i][s])
        standard_deviation = np.std(inference_times[i][s])
        mean_inference_times.append(mean)
        print("Mean inference time for " + trained_section_names[s] +  " of machine type '" + machine_types[i] + "' is " + str(mean) + " s "
                "with standard deviation of " + str(standard_deviation))
