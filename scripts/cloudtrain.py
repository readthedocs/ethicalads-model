"""
Trains our ML model in the cloud.

This script does the following:

- Spins up a new LambdaLabs GPU instance
- Installs our model and prereqs
- Trains the model with SpaCy
- Copies the built model to the packages/ directory
- Terminates the GPU instance

The script has a number of command line options. See --help.

Uses the LambdaLabs API: https://cloud.lambdalabs.com/api/v1/docs
You must set the envvar $LAMBDALABS_KEY to use this script.
"""

import datetime
import os
import subprocess
import time

import requests


PACKAGES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../packages"))

LAMBDA_API_KEY = os.environ.get("LAMBDALABS_KEY")
if not LAMBDA_API_KEY:
    raise RuntimeError("Environment variable $LAMBDALABS_KEY not set")

DEFAULT_SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_rsa.pub")
LAMBDALABS_API_PATH = "https://cloud.lambdalabs.com/api/v1/"

# We will start the first instance possible matching one of these
DESIRED_INSTANCE_TYPES = (
    "gpu_1x_rtx6000",  # ~50c/hr
    "gpu_1x_a6000",  # ~80c/hr
)


def lambdalabs_api_call(path, method="GET", json=None):
    resp = requests.request(
        method,
        "https://cloud.lambdalabs.com/api/v1" + path,
        auth=(LAMBDA_API_KEY, ""),
        json=json,
    )

    if not resp.ok:
        print("LambdaLabs API call unexpectedly failed.")
        print(resp.content)
        resp.raise_for_status()

    return resp


def get_available_instance():
    instance_type_name = None
    region_name = None

    # Get which of our desired instance types has capacity
    offered_instances = get_offered_instances()
    for name in offered_instances:
        if name not in DESIRED_INSTANCE_TYPES:
            continue

        instance_type = offered_instances[name]

        # Use the first available region
        if instance_type["regions_with_capacity_available"]:
            region_name = instance_type["regions_with_capacity_available"][0]["name"]
            instance_type_name = name
            break

    # Returns None, None if nothing is available
    return instance_type_name, region_name


def get_offered_instances():
    resp = lambdalabs_api_call("/instance-types")
    return resp.json()["data"]


def get_running_instances():
    resp = lambdalabs_api_call("/instances")
    return resp.json()["data"]


def get_ssh_keys():
    resp = lambdalabs_api_call("/ssh-keys")
    return [key["name"] for key in resp.json()["data"]]


def launch_instance(region_name, instance_type_name, ssh_key_name):
    # This name is simply a helpful name attached to the instance
    now = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    name = f"ethicalads_model_trainer_{now}"

    resp = lambdalabs_api_call(
        "/instance-operations/launch",
        method="POST",
        json={
            "region_name": region_name,
            "instance_type_name": instance_type_name,
            "ssh_key_names": [ssh_key_name],  # Currently this cannot accept multiple,
            "quantity": 1,
            "name": name,
        },
    )

    # Returns a list of 1 item since the quantity of instances is 1
    instances = resp.json()["data"]["instance_ids"]

    # Return the instance ID (a string)
    # We need this ID to terminate the instance
    return instances[0]


def get_instance_details(instance_id):
    resp = lambdalabs_api_call("/instances/" + instance_id)
    return resp.json()["data"]


def wait_for_instance(instance_id):
    # Loop until the instance has booted up and is available
    while True:
        time.sleep(5)
        details = get_instance_details(instance_id)
        if details["status"] == "active":
            break


def run_ssh_command(ssh_identity_file, instance_ip, command):
    print(command)
    subprocess.check_call(
        [
            "ssh",
            "-i",
            ssh_identity_file,
            "-l",
            "ubuntu",
            # Accept the host's fingerprint
            "-o",
            "StrictHostKeyChecking=accept-new",
            instance_ip,
            command,
        ]
    )


def train_model(ssh_identity_file, instance_ip):
    commands = [
        "mkdir -p ~/checkouts && git clone https://github.com/readthedocs/ethicalads-model.git ~/checkouts/ethicalads-model",
        "cd ~/checkouts/ethicalads-model && pip install -r requirements.txt",
        # Debugging commands that deal with GPUs
        # and whether the CUDA and GPU libs are setup correctly
        "nvidia-smi",
        "python -c 'import torch; print(torch.cuda.is_available())'",
        "python -c 'import cupy; import cupyx; print(cupy.cuda.runtime.getDeviceCount())'",
        "python -c 'import spacy; print(spacy.require_gpu())'",
        # End debugging commands
        # Actually train the model
        "cd ~/checkouts/ethicalads-model && python scripts/generate-training-test-sets.py -o assets/train.json -f assets/test.json assets/categorized-data.yml",
        "cd ~/checkouts/ethicalads-model && python -m spacy project run all . --vars.train=train --vars.dev=test --vars.name=ethicalads_topics --vars.version=`date '+%Y%m%d_%H_%M_%S'`",
    ]

    for command in commands:
        run_ssh_command(
            ssh_identity_file,
            instance_ip,
            command,
        )


def copy_trained_model(ssh_identity_file, instance_ip):
    subprocess.check_call(
        [
            "scp",
            "-i",
            ssh_identity_file,
            # Accept the host's fingerprint
            "-o",
            "StrictHostKeyChecking=accept-new",
            f"ubuntu@{instance_ip}:~/checkouts/ethicalads-model/packages/en_ethicalads_topics*/dist/en_ethicalads_topics-*.tar.gz",
            PACKAGES_DIR,
        ],
    )


def terminate_instance(instance_id):
    resp = lambdalabs_api_call(
        "/instance-operations/terminate",
        method="POST",
        json={
            "instance_ids": [instance_id],  # We are just terminating a single instance
        },
    )


def main(args):
    ssh_identity_file = args.ssh_identity_file
    ssh_key_name = args.ssh_key_name

    # Get the SSH key to use
    if not ssh_key_name:
        keys = get_ssh_keys()
        for key in keys:
            ssh_key_name = key
            print(f"No SSH key name specified. Using {ssh_key_name}.")
            break

    instance_type_name, region_name = get_available_instance()
    if args.wait_for_capacity and (not instance_type_name or not region_name):
        print("Waiting for GPU instances to become available...")
        while (not instance_type_name or not region_name):
            time.sleep(15)
            instance_type_name, region_name = get_available_instance()

    if not instance_type_name or not region_name:
        # No GPU instance capacity
        print("No GPU capacity available of desired instance types. Quitting...")
        return

    print(
        f"Launching instance {instance_type_name} on region {region_name} with ssh key {ssh_key_name}..."
    )
    instance_id = launch_instance(region_name, instance_type_name, ssh_key_name)

    print("Waiting for instance...")
    wait_for_instance(instance_id)
    print("Instance is active!")

    details = get_instance_details(instance_id)
    instance_ip = details["ip"]

    print(f"Training model on {instance_ip}...")
    print("*" * 77)
    train_model(ssh_identity_file, instance_ip)
    print("*" * 77)

    print("Copying trained model to local packages/ directory...")
    copy_trained_model(ssh_identity_file, instance_ip)

    if not args.skip_termination:
        print(f"Terminating instance {instance_id}...")
        terminate_instance(instance_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Trains our ML model in the cloud on LambdaLabs' GPU instances",
    )
    parser.add_argument(
        "-i",
        "--ssh-identity-file",
        help="Path to the SSH public key used to connect to the cloud instance [~/.ssh/id_rsa.pub]",
        default=DEFAULT_SSH_KEY_PATH,
    )
    parser.add_argument(
        "--ssh-key-name",
        help="LambdaLabs SSH Key Name to use (defaults to the first one on https://cloud.lambdalabs.com/ssh-keys)",
        default=None,
    )
    parser.add_argument(
        "--skip-termination",
        help="Leave the instance running (and billing) after training",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--wait-for-capacity",
        help="Wait for GPU instances to become available",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    try:
        main(args)
    finally:
        instances = get_running_instances()
        if len(instances) == 0:
            print("👍 There are no currently running instances.")
        else:
            print(instances)
            print("IMPORTANT: There are currently running instances!!")
            print("  You are responsible for shutting these down.")
            print("  These are being billed at an hourly rate!")
