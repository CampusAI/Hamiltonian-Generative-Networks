# Hamiltonian-Generative-Networks
[![DOI](https://zenodo.org/badge/295400716.svg)](https://zenodo.org/badge/latestdoi/295400716)

Re-implementation of Hamiltonian Generative Networks [paper](https://arxiv.org/abs/1909.13789).
You can find the re-implementation publication details [here](https://rescience.github.io/bibliography/Balsells_Rodas_2021.html), and article [here](https://zenodo.org/record/4835278#.YQUhI3UzaEC).


## Setup

1. Install (CPU or GPU) [PyTorch](https://pytorch.org/). Tested on 1.6.0
2. Install other project dependencies:
`pip install -r requirements.txt`

## Modules

- **[Environments](environments/)**: You can find the implementation of the physical environments used to get the ground truth data.

- **[Networks](networks/)**: Contains the definitions of the main ANNs used: Encoder, Transformer, Hamiltonian, and Decoder.

- **[Utilities](utilities/)**: Holds helper classes such as the HGN integrator and a HGN output bundler class.

- **[Experiment Params](experiment_params/)**: Contains .yaml files with the meta-parameters used in different experiments.

- **[hamiltonian_generative_network.py](hamiltonian_generative_network.py)** script contains the definition of the HGN architecture.

## How to train
The [train.py](train.py) script takes care of performing the training.
To start a training, run 
```commandline
python train.py --train-config <path_to_train_config_file> --dataset-config <path_to_data_config_file>
```

```
optional arguments:
  -h, --help            show this help message and exit
  --train-config TRAIN_CONFIG
                        Path to the training configuration yaml file.
  --dataset-config DATASET_CONFIG
                        Path to the dataset configuration yaml file.
  --name NAME           If specified, this name will be used instead of
                        experiment_id of the yaml file.
  --epochs EPOCHS       The number of training epochs. If not specified,
                        optimization.epochs of the training configuration will
                        be used.
  --env ENV             The environment to use (for online training only).
                        Possible values are 'pendulum', 'spring',
                        'two_bodies', 'three_bodies', corresponding to
                        environment configurations in
                        experiment_params/default_environments/. If not
                        specified, the environment specified in the given
                        --dataset-config will be used.
  --dataset-path DATASET_PATH
                        Path to a stored dataset to use for training. For
                        offline training only. In this case no dataset
                        configuration file will be loaded.
  --params PARAMS [PARAMS ...]
                        Override one or more parameters in the config. The
                        format of an argument is param_name=param_value.
                        Nested parameters are accessible by using a dot, i.e.
                        --param dataset.img_size=32. IMPORTANT: lists must be
                        enclosed in double quotes, i.e. --param
                        environment.mass:"[0.5, 0.5]".
  --resume [RESUME]     NOT IMPLEMENTED YET. Resume the training from a saved
                        model. If a path is provided, the training will be
                        resumed from the given checkpoint. Otherwise, the last
                        checkpoint will be taken from
                        saved_models/<experiment_id>.
```
The `experiment_params/` folder contains default dataset and training configuration files.
Training can be done in on-line or off-line mode.

- In **on-line mode** data is generated during training, eliminating the need for a
heavy dataset. A dataset configuration file must be provided in the `--dataset-config`
argument. This file must define the `environment:` and `dataset:` sections
(see [experiment_params/dataset_online_default.yaml](experiment_params/dataset_online_default.yaml))
. The `--env` argument may be used to override the environment defined in the config file
with one of the default environments in `experiment_params/default_environments/`.

- In **off-line mode** the training is performed from a stored dataset (see the section below
on how to generate datasets). A dataset config specifying the train and test dataset paths
in the `train_data:` and `test_data:` sections can be given to `--dataset-config` (see
[experiment_params/dataset_offline_default.yaml](experiment_params/dataset_offline_default.yaml))
. Otherwise, the path to an existing dataset root folder (the one containing the
`parameters.yaml` file ) must be provided to the `--dataset-path` argument. 
## Generating and saving datasets
A dataset can be generated starting from a `yaml` parameter file that specifies all its parameters
in the `environment` and `dataset` sections. To create a dataset, run
```commandline
python generate_data.py
```
which will create the dataset in a folder with the given name (see args below) and will
write a `parameters.yaml` file within it, that can be directly used for off-line training
on the created dataset.

```
optional arguments:
  -h, --help            show this help message and exit
  --name NAME           The dataset name.
  --dataset-config DATASET_CONFIG
                        YAML file from which to read the dataset parameters.
                        If not specified,
                        experiment_params/dataset_online_default.yaml will be
                        used.
  --ntrain NTRAIN       Number of training sample to generate.
  --ntest NTEST         Number of test samples to generate.
  --env ENV             The default environment specifications to use. Can be
                        'pendulum', 'spring', 'two_bodies', 'three_bodies',
                        'chaotic_pendulum'. If this argument is specified, a
                        default environment section will be loaded from the
                        correspondent yaml file in
                        experiment_params/default_environments/
  --datasets-root DATASETS_ROOT
                        Root of the datasets folder in which the dataset will
                        be stored. If not specified, datasets/ will be used as
                        default.
  --params PARAMS [PARAMS ...]
                        Override one or more parameters in the config. The
                        format of an argument is param_name=param_value.
                        Nested parameters are accessible by using a dot, i.e.
                        --param dataset.img_size=32. IMPORTANT: lists must be
                        enclosed in double quotes, i.e. --param
                        environment.mass:"[0.5, 0.5]".
```
