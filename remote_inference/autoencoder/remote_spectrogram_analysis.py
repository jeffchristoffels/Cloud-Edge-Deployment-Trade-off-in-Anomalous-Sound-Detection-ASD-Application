
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
data_list = request["data_list"]
data = np.array(data_list)

# Analysis
param = com.yaml_load()
with open("directories.yaml") as stream:
    directories = yaml.safe_load(stream)

model_path = "{model_directory}/autoencoder/baseline/model_{machine_type}.hdf5" \
    .format(model_directory=directories["model_directory"],
            machine_type=machine_type)
model = tf.keras.models.load_model(model_path)

# load anomaly score distribution for determining threshold
score_distr_file_path = "{model_directory}/autoencoder/baseline/score_distr_{machine_type}.pkl" \
    .format(model_directory=directories["model_directory"],
            machine_type=machine_type)
shape_hat, loc_hat, scale_hat = joblib.load(score_distr_file_path)

# determine threshold for decision
decision_threshold = scipy.stats.gamma.ppf(q=param["decision_threshold"], a=shape_hat, loc=loc_hat, scale=scale_hat)

y_pred = np.mean(np.square(data - model.predict(data)))

if y_pred > decision_threshold:
    analysis_result = "anomalous"
else:
    analysis_result = "normal"

# Send the analysis result back to the local machine through standard output (stdout)
sys.stdout.buffer.write(analysis_result.encode())
