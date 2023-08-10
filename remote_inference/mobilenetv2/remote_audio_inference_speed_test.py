
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

print("Remote audio inference speed test for baseline mobilenetv2")

def run_remote_analysis(remote_ip, username, private_key_path, dir, audio_file, machine_type, section_name):
    # Copy audio to GPU server
    scp_command =["scp",
                  "-i",
                  private_key_path,
                  f"{dir}/{audio_file}",
                  f"{username}@{remote_ip}:~/remote_inference/mobilenetv2/remote_audio"]
    try:
        # Run the SCP command
        subprocess.run(scp_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

    # Execute analysis script and remove audio
    ssh_command = [
        "ssh",
        "-i",
        private_key_path,
        f"{username}@{remote_ip}",
        f"source ~/miniconda3/bin/activate {conda_environment} &&"
        f"cd ~/remote_inference/mobilenetv2 &&"
        f"python remote_audio_analysis.py &&"
        f"rm ~/remote_inference/mobilenetv2/remote_audio/{audio_file}"
    ]

    # Combine all data and send it over SSH
    combined_data = json.dumps({"audio_file": audio_file,
                                "machine_type": machine_type,
                                "section_name": section_name})

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

        section_name = '_'.join(audio_file.split('_')[:2])

        run_remote_analysis(remote_ip, username, private_key_path, dir, audio_file, machine_type, section_name)

        current_inference_times.append(time.time() - start_time)

    inference_times.append(current_inference_times)

print("\n\n")
for i in range(len(inference_times)):
    average_time = np.mean(inference_times[i])
    standard_deviation = np.std(inference_times[i])
    print("Mean inference time for machine type '" + machine_types[i] + "' is " + str(average_time) + " s "
            "with standard deviation of " + str(standard_deviation))