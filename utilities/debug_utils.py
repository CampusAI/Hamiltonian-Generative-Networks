import yaml
import sys
import os

from matplotlib import pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks import inference_net
import train


torch.autograd.set_detect_anomaly(True)


GRADIENTS = {}  # Each element is [counter, grad]


def backward_hook(module, grad_input, grad_output):
    """Hook called in the backward pass of modules, saving the gradients in the GRADIENTS dict.

    Args:
        module (torch.nn.Module): The module for which this backward pass is called. Must provide a
            'name' attribute, that will be used as key in the GRADIENTS dict.
        grad_input (tuple): Tuple (dL/dx, dL/dw, dL/db)
        grad_output (tuple): 1-tuple (dL/do) i.e. the gradient of the loss w.r.t. the layer output.
    """
    if module.name == 'Transformer_out':
        q, p = inference_net.TransformerNet.to_phase_space(grad_output[0])
        set_gradient('Transformer_out_q', q.detach().numpy())
        set_gradient('Transformer_out_p', p.detach().numpy())
    else:
        set_gradient(module.name, grad_output[0].detach().numpy())
    return None


def set_gradient(name, gradient):
    if name in GRADIENTS:
        GRADIENTS[name][0] += 1
        GRADIENTS[name][1] += gradient
    else:
        GRADIENTS[name] = [1, gradient]


def register_hooks(hgn):
    """Set a name to all the interesting layers of the hamiltonian generative networks and register
    hook.

    Args:
        hgn (hamiltonian_generative_network.HGN): The HGN to analyse.
    """
    # Setting name variable to be used in hook
    hgn.encoder.input_conv.name = 'Encoder_in'
    hgn.encoder.out_mean.name = 'Encoder_out_mean'
    hgn.encoder.out_logvar.name = 'Encoder_out_logvar'
    hgn.transformer.in_conv.name = 'Transformer_in'
    hgn.transformer.out_conv.name = 'Transformer_out'
    hgn.hnn.in_conv.name = 'Hamiltonian_in'
    hgn.hnn.linear.name = 'Hamiltonian_out'
    hgn.decoder.residual_blocks[0].name = 'Decoder_in'
    hgn.decoder.out_conv.name = 'Decoder_out'

    # Registering hooks
    hgn.encoder.input_conv.register_backward_hook(backward_hook)
    hgn.encoder.out_mean.register_backward_hook(backward_hook)
    hgn.encoder.out_logvar.register_backward_hook(backward_hook)
    hgn.transformer.in_conv.register_backward_hook(backward_hook)
    hgn.transformer.out_conv.register_backward_hook(backward_hook)
    hgn.hnn.in_conv.register_backward_hook(backward_hook)
    hgn.hnn.linear.register_backward_hook(backward_hook)
    hgn.decoder.residual_blocks[0].register_backward_hook(backward_hook)
    hgn.decoder.out_conv.register_backward_hook(backward_hook)


def plot_grads(hgn, batch_size, dtype):
    """Plot the gradients of each input-output layer of the hamiltonian generative network model.

    Args:
        hgn (hamiltonian_generative_network.HGN): The HGN to analyze.
        batch_size (int): Batch size used when testing gradients
        dtype (torch.dtype): Type to be used in tensor operations.

    """
    register_hooks(hgn)
    rand_in = torch.rand((batch_size, hgn.seq_len, hgn.channels, 32, 32)).type(dtype)
    hgn.fit(rand_in)

    for k, v in GRADIENTS.items():
        print(k + ' backward called ' + str(v[0]) + ' times')
    print('------------------------------------------------')

    names = GRADIENTS.keys()
    max_grads = [np.abs((GRADIENTS[k][1] / GRADIENTS[k][0])).max() for k in names]
    mean_grads = [np.abs((GRADIENTS[k][1] / GRADIENTS[k][0])).mean() for k in names]

    for name, max_grad, mean_grad in zip(names, max_grads, mean_grads):
        print(name + ' max_grad: ' + str(max_grad) + ' mean_grad: ' + str(mean_grad))

    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), mean_grads, alpha=0.3, lw=1, color="b")
    plt.hlines(0, 0, len(mean_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(mean_grads), 1), names, rotation="vertical")
    plt.xlim(left=0, right=len(mean_grads))
    plt.ylim(bottom=-0.000001, top=0.0001)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)],
               ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()


if __name__ == '__main__':
    params_file = "experiment_params/default.yaml"
    with open(params_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    device = "cuda:" + str(params["gpu_id"]) if torch.cuda.is_available() else "cpu"

    hgn = train.load_hgn(params, device=device, dtype=torch.float)

    plot_grads(hgn, batch_size=params['optimization']['batch_size'], dtype=torch.float)