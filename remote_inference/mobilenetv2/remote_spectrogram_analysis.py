
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Only print messages that are more important, while ignoring less critical messages

import sys
import json
import yaml
import tensorflow as tf
import common as com

import numpy as np
import scipy.stats

try:
    from sklearn.externals import joblib
except:
    import joblib

# Read the input data from standard input (stdin)
input_data = sys.stdin.buffer.read().decode()
request = json.loads(input_data)

# Split input data
machine_type = request["machine_type"]
section_name = request["section_name"]
data_list = request["data_list"]
data = np.array(data_list)

# Analysis
param = com.yaml_load()
with open("directories.yaml") as stream:
    directories = yaml.safe_load(stream)

model_path = "{model_directory}/mobilenetv2/baseline/model_{machine_type}.hdf5" \
    .format(model_directory = directories["model_directory"],
            machine_type = machine_type)
model =  tf.keras.models.load_model(model_path)

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

temp_array = np.nonzero(trained_section_names == section_name)[0]
section_idx = temp_array[0]

# make one-hot vector for conditioning
condition = np.zeros((data.shape[0], n_sections), float)
condition[:, section_idx: section_idx + 1] = 1

# 1D vector to 2D image
data = data.reshape(data.shape[0], param["feature"]["n_frames"], param["feature"]["n_mels"], 1)

p = model.predict(data)[:, section_idx: section_idx + 1]
y_pred = np.mean(np.log(np.maximum(1.0 - p, sys.float_info.epsilon)
                        - np.log(np.maximum(p, sys.float_info.epsilon))))

if y_pred > decision_threshold:
    analysis_result = "anomalous"
else:
    analysis_result = "normal"

# Send the analysis result back to the local machine through standard output (stdout)
sys.stdout.buffer.write(analysis_result.encode())
