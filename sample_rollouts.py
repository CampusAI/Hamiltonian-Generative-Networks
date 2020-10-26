"""This script shows how to sample new rollouts from a variational-trained HGN.
"""
from hamiltonian_generative_network import HGN
from utilities.integrator import Integrator

if __name__=="__main__":
    model_to_load = "saved_models/two_bodies_default"
    
    integrator = Integrator(delta_t=0.125, method="Leapfrog")
    hgn = HGN(integrator=integrator)  # If going to load, no need to specify networks
    hgn.load(model_to_load)
    
    # Sample a rollout of n_steps
    prediction = hgn.get_random_sample(n_steps=50, img_shape=(32, 32))
    prediction.visualize()
    
