# Machine Learning for Ads!

This is the machine learning classifier used by [EthicalAds](https://ethicalads.io).
It classifies content into the major topics targeted by advertisers on EthicalAds.

This project uses [spaCy](https://spacy.io) to do text classification around text for ad targeting.


## Quickstart

This will generate our training data and then build and train the model.

    # Generate training and test set from the categorized data (Yaml file)
    python scripts/generate-training-test-sets.py -o assets/train.json -f assets/test.json assets/categorized-data.yml
    python -m spacy project run all . --vars.train=train --vars.dev=test --vars.name=ethicalads_topics --vars.version=`date "+%Y%m%d_%H_%M_%S"`

### Running the analyzer

After installing the analyzer (it's installed in staging already),
you can run it against an arbitrary URL to see how that page was classified.
Note: this command is run inside the main project [ethical-ad-server](https://github.com/readthedocs/ethical-ad-server).

    ADSERVER_ANALYZER_BACKEND=adserver.analyzer.backends.EthicalAdsTopicsBackend ./manage.py runmodel https://example.com


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


## ☁️ Building in the Cloud

Currently, we're building our model on [LamdbaLabs Cloud GPU instances](https://lambdalabs.com/).

    # ssh into your instance
    sudo apt-get update
    sudo apt-get install -y curl wget sudo git

    mkdir -p ~/checkouts
    git clone https://github.com/readthedocs/ethicalads-model.git ~/checkouts/ethicalads-model
    cd ~/checkouts/ethicalads-model
    pip install -r requirements.txt

    # Shows info for GPUs
    nvidia-smi

    # These should not error
    python -c 'import torch; print(torch.cuda.is_available())'
    python -c 'import cupy; import cupyx; print(cupy.cuda.runtime.getDeviceCount())'
    python -c 'import spacy; print(spacy.require_gpu())'

    # When these are complete, your model will be in packages/
    python scripts/generate-training-test-sets.py -o assets/train.json -f assets/test.json assets/categorized-data.yml | tee -a modelbuild.out
    python -m spacy project run all . --vars.train=train --vars.dev=test --vars.name=ethicalads_topics --vars.version=`date "+%Y%m%d_%H_%M_%S"` | tee -a modelbuild.out


## 📚 Data

Our data is hand-labeled URL's that are located in ``assets/categorized-data.yml``.
This maps a specific URL to a topic,
and then we download the data from those URL's and split them into a training & validation set with ``scripts/generate-training-test-sets.py``.


## Deployment

We are currently just uploading a zipfile of the Python model,
and then installing it in our deployment scripts into a baked build image.

This can be found in our closed source ``ethicalads-ops`` repo that has custom deployment code.
