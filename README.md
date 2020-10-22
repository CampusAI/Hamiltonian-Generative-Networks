# Hamiltonian-Generative-Networks
Re-implementation of Hamiltonian Generative Networks [paper](https://arxiv.org/abs/1909.13789)


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
```cmd
python train.py [--params param_file] [--name experimen_name] [--cpu]
```
where the optional argument `--param` can be used to specify the parameter
file to be used. `--name` can be used to overwrite the `experiment_id` of the
yaml file and save the data under the new name. `--cpu` can be used to force
training on the CPU, otherwise the training will be performed in the GPU (if available). 

Training can be done in on-line or off-line mode.

- In on-line mode the dataset is generated during training. The parameter file must therefore
contain a `environment:` section with all the parameters of the environment to use.
The parameter `dataset:` section must contain the parameters for dataset generation.
See [experiment_params/default_online.yaml](experiment_params/default_online.yaml) for an
exhaustive example.
- In off-line mode the training is performed on a saved dataset. Therefore, the
`dataset:` section of the parameter file must only contain the training data and test data paths
in the corrsepondent variables `train_data` and `test_data`.
See [experiment_params/default_offline](experiment_params/default_offline.yaml) for an exhaustive
example.

## Generating and saving datasets
A dataset can be generated starting from a `yaml` parameter file that specifies all its parameters
in the `environment` and `dataset` sections. To create a dataset, run
```commandline
python environments/generate_data.py

Creates a train and test dataset and prints a .yaml file ready to be run for offline training.
```

```commandline
Optional arguments:
  -h, --help            show this help message and exit
  --params PARAMS       YAML file from which to read the dataset parameters.
                        If not specified,experiment_params/default_online.yaml
                        will be used.
  --name NAME           Use this name for the dataset instead of
                        experiment_name in the yaml file.
  --ntrain NTRAIN       Number of training sample to generate.
  --ntest NTEST         Number of test samples to generate.
  --env ENV             The default environment specifications to use. Can be
                        'pendulum', 'spring', 'two_bodies', 'three_bodies',
                        'chaotic_pendulum'. If this argument is specified, a
                        default environment section will be loaded from the
                        correspondent yaml file in
                        experiment_params/default_environments/
  --env-spec ENV_SPEC [ENV_SPEC ...]
                        Parameters of the environment in the form
                        param_name:param_value, e.g. --env-spec g:1.0
                        mass:0.5. If this argument is specified, the given
                        parameters will be used instead of those in the yaml
                        file.
  --datasets-root DATASETS_ROOT
                        Root of the datasets folder in which the dataset will
                        be stored. If not specified, datasets/ will be used as
                        default.
```

**Important:** The given parameter file for dataset generation must fully specify the `dataset` and
 `environment` sections. The parameter file will be then saved into the created dataset folder, 
 converted to the offline parameters.