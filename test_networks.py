import numpy as np
import torch

from networks import hamiltonian_net
from networks import inference_network


if __name__ == '__main__':
    SEQUENCE_LENGTH = 10
    BATCH_SIZE = 64
    enc_net = inference_network.EncoderNet(seq_len=SEQUENCE_LENGTH)

    rand_images = np.random.randint(0, 255, size=(BATCH_SIZE, SEQUENCE_LENGTH, 32, 32))
    rand_images_ts = torch.tensor(rand_images).float()

    z = enc_net(rand_images_ts)

    trans_net = inference_network.TransformerNet(in_channels=48, out_channels=32)
    encoding = trans_net(z)

    ham_net = hamiltonian_net.HamiltonianNet(in_channels=32)

    h = ham_net(encoding)
    print(h.shape)