
import yaml
import argparse
import os, random, time
import json
import subprocess

import common as com
import numpy as np

########################################################################
# load information
########################################################################

param = com.yaml_load()
with open("directories.yaml") as stream:
    directories = yaml.safe_load(stream)

remote_ip = param["ssh"]["remote_ip"]
username = param["ssh"]["username"]
private_key_path = param["ssh"]["private_key_path"]
conda_environment = param["ssh"]["conda_environment"]

# Parse arguments of command line
parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')
parser.add_argument("audio_amount")
args = parser.parse_args()

audio_amount = int(args.audio_amount)

########################################################################

print("Remote spectrogram inference speed test for baseline autoencoder")

def run_remote_analysis(remote_ip, username, private_key_path, data, machine_type):
    ssh_command = [
        "ssh",
        "-i",
        private_key_path,
        f"{username}@{remote_ip}",
        f"source ~/miniconda3/bin/activate {conda_environment} &&"
        f"cd ~/remote_inference/autoencoder &&"
        f"python ./remote_spectrogram_analysis.py"
    ]

    # Combine all data and send it over SSH
    data_list = data.tolist()
    combined_data = json.dumps({"data_list": data_list,
                                "machine_type": machine_type})

    process = subprocess.Popen(ssh_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    analysis_result, _ = process.communicate(input=combined_data.encode())

    # 'analysis_result' contains the output returned by the remote analysis script
    print("Remote analysis result:")
    print(analysis_result.decode())

inference_times = []

machine_types = ["fan", "gearbox", "pump", "slider", "valve", "ToyCar", "ToyTrain"]
for machine_type in machine_types:      # Test all machine types
    print("\n\nMachine type: " + machine_type + "\n")

    current_inference_times = []

    # Pick audio_amount random target test audio files
    dir = "{audio_directory}/{machine_type}/target_test/"\
        .format(audio_directory = directories["audio_directory"],
                machine_type = machine_type)
    filenames = random.sample(os.listdir(dir), audio_amount)

    for audio_file in filenames:

        start_time = time.time()

        audio_path = dir + audio_file
        try:
            data = com.file_to_vectors(audio_path,
                                       n_mels=param["feature"]["n_mels"],
                                       n_frames=param["feature"]["n_frames"],
                                       n_fft=param["feature"]["n_fft"],
                                       hop_length=param["feature"]["hop_length"],
                                       power=param["feature"]["power"])
        except:
            com.logger.error("File broken or doesn't exist: {}".format(audio_path))

        run_remote_analysis(remote_ip, username, private_key_path, data, machine_type)

        current_inference_times.append(time.time() - start_time)

    inference_times.append(current_inference_times)

print("\n\n")
for i in range(len(inference_times)):
    average_time = np.mean(inference_times[i])
    standard_deviation = np.std(inference_times[i])
    print("Mean inference time for machine type '" + machine_types[i] + "' is " + str(average_time) + " s "
            "with standard deviation of " + str(standard_deviation))