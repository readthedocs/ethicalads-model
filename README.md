# Machine Learning for Ads!

This is the machine learning classifier used by [EthicalAds](https://ethicalads.io).
It classifies content into the major topics targeted by advertisers on EthicalAds.

This project uses [spaCy](https://spacy.io) to do text classification around text for ad targeting.


## Quickstart

This will generate our training data and then build and train the model locally.

    # Generate training and test set from the categorized data (Yaml file)
    python scripts/generate-training-test-sets.py -o assets/train.json -f assets/test.json assets/categorized-data.yml
    python -m spacy project run all . --vars.train=train --vars.dev=test --vars.name=ethicalads_topics --vars.version=`date "+%Y%m%d_%H_%M_%S"`


## ☁️ Training in the Cloud

Training is done best with a GPU.
Currently, we're training our model on [LamdbaLabs Cloud GPU instances](https://lambdalabs.com/).

    export LAMBDALABS_KEY=xxxxxxxxxxx
    python scripts/cloudtrain.py

This script will spin up a cloud GPU instance, train the model, and teardown the instance.
The resulting model will be copied to the local `packages/` directory.


## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

For training with a GPU, some modifications to the `project.yml` are needed.
Specifically, set the `gpu_id` (to 0 usually) and the `config` to `gpu-efficiency.cfg`.

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `preprocess` | Convert the data to spaCy's binary format |
| `train` | Train a text classification model |
| `evaluate` | Evaluate the model and export metrics |
| `package` | Build the actual Python package for the model to install |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `preprocess` &rarr; `train` &rarr; `evaluate` |


## 📚 Data

Our data is hand-labeled URL's that are located in ``assets/categorized-data.yml``.
This maps a specific URL to a topic,
and then we download the data from those URL's and split them into a training & validation set with ``scripts/generate-training-test-sets.py``.


## Deployment

We are currently just uploading a zipfile of the Python model,
and then installing it in our deployment scripts into a baked build image.

This can be found in our closed source ``ethicalads-ops`` repo that has custom deployment code.
