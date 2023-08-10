
import os, random, time
import tensorflow as tf
# import tflite_runtime.interpreter as tflite
import gc
import argparse
import yaml

start_time = time.time()

########################################################################
# import additional libraries
########################################################################
import numpy as np
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
parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')
parser.add_argument("optimization")
parser.add_argument("audio_amount")
args = parser.parse_args()

optimization = args.optimization
audio_amount = int(args.audio_amount)

########################################################################

print("Inference speed test for compressed autoencoder with " + optimization)
print("Time needed for imports: " + str(time.time() - start_time) + " s")

########################################################################################################################

machine_types = ["fan", "gearbox", "pump", "slider", "valve", "ToyCar", "ToyTrain"]
started_inference = False       # Don't take into account first inference time
inference_times = []
decision_thresholds = []
anomaly_scores = []
decision_results = []

for machine_type in machine_types:      # Test all machine types

    start_time = time.time()

    model_path = "{model_directory}/autoencoder/{optimization}/model_{machine_type}.tflite"\
        .format(model_directory = directories["model_directory"],
                optimization = optimization,
                machine_type = machine_type)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    input_details = interpreter.get_input_details()
    interpreter.resize_tensor_input(0, [309, 640], strict=True)

    # load anomaly score distribution for determining threshold
    score_distr_file_path = "{model_directory}/autoencoder/{optimization}/score_distr_{machine_type}.pkl"\
        .format(model_directory=directories["model_directory"],
                optimization=optimization,
                machine_type=machine_type)
    shape_hat, loc_hat, scale_hat = joblib.load(score_distr_file_path)

    # determine threshold for decision
    decision_threshold = scipy.stats.gamma.ppf(q=param["decision_threshold"], a=shape_hat, loc=loc_hat, scale=scale_hat)
    decision_thresholds.append(decision_threshold)

    # Pick 100 random target test audio files
    dir = "{audio_directory}/{machine_type}/target_test/"\
        .format(audio_directory = directories["audio_directory"],
                machine_type = machine_type)

    filenames = random.sample(os.listdir(dir), audio_amount + 1)

    current_inference_times = []
    current_anomaly_scores = []
    current_decision_results = []

    print("Loaded data for " + machine_type + " in " + str(time.time() - start_time) + " s")

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
            com.logger.error("File broken!!: {}".format(file_path))

        input_data = np.array(data, dtype = input_details[0]['dtype'])
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        output_details = interpreter.get_output_details()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        y_pred = np.mean(np.square(input_data - output_data))

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

        current_inference_times.append(time.time() - start_time)

    inference_times.append(current_inference_times)
    anomaly_scores.append(current_anomaly_scores)
    decision_results.append(current_decision_results)

    del data
    del interpreter
    gc.collect()

average_inference_times = []
for i in range(len(inference_times)):
    average_time = np.mean(inference_times[i])
    standard_deviation = np.std(inference_times[i])
    average_inference_times.append(average_time)
    print("Mean inference time for machine type '" + machine_types[i] + "' is " + str(average_time) + " s "
            "with standard deviation of " + str(standard_deviation))
