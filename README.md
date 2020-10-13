# Hamiltonian-Generative-Networks
Re-implementation of Hamiltonian Generative Networks [paper](https://arxiv.org/abs/1909.13789)


## Setup

1. Install (CPU or GPU) [PyTorch](https://pytorch.org/) Tested on 1.6.0
2. Install other project dependencies:
`pip install -r requirements.txt`

## Modules

To train the HGN, run [train.py](train.py)

- **[Environments](environments/)**: You can find the implementation of the physical environments used to get the ground truth data.

- **[Networks](networks/)**: Contains the definitions of the main ANNs used: Encoder, Transformer, Hamiltonian, and Decoder.

- **[Utilities](utilities/)**: Holds helper classes such as the HGN integrator and a HGN output bundler class.

- **[Experiment Params](experiment_params/)**: Contains .yaml files with the meta-parameters used in different experiments.

- **[hamiltonian_generative_network.py](hamiltonian_generative_network.py)** script contains the definition of the HGN architecture.