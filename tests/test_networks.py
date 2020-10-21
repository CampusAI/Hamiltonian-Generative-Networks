import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path('.').absolute().parent))
from networks import encoder_net
from networks import hamiltonian_net
from networks import transformer_net

if __name__ == '__main__':
    SEQUENCE_LENGTH = 10
    BATCH_SIZE = 64
    enc_net = encoder_net.EncoderNet(seq_len=SEQUENCE_LENGTH, out_channels=48)

    rand_images = np.random.randint(0,
                                    255,
                                    size=(BATCH_SIZE, SEQUENCE_LENGTH, 32, 32))
    rand_images_ts = torch.tensor(rand_images).float()

    z, mean, stdev = enc_net(rand_images_ts)

    trans_net = transformer_net.TransformerNet(in_channels=48, out_channels=32)
    q, p = trans_net(z)

    ham_net = hamiltonian_net.HamiltonianNet(in_channels=32)

    h = ham_net(q, p)
