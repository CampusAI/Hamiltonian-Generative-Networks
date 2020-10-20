""" Iterates through all the yaml files of the given directory and trains a model
"""
import os
import sys

import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import train

if __name__=="__main__":
    base_params_file = "experiment_params/overfit_test.yaml"
    train_params_dir = "experiment_params/overfit_grid_search/"
    
    for file in os.listdir(train_params_dir):
        if file.endswith(".yaml"):
            with open(base_params_file, 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
        
            with open(os.path.join(train_params_dir, file), 'r') as f:
                params_to_update = yaml.load(f, Loader=yaml.FullLoader)

            params.update(params_to_update)
            train(params)
