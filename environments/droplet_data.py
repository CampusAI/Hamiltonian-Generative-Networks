import os
import sys
import cv2

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import median_filter


class DroplerLoader(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)
        

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        path = os.path.join(self.root_dir, self.file_list[i])
        image_list = sorted(os.listdir(path))
        im_list = []
        for i in range(len(image_list)//20):
            file_path = os.path.join(path, image_list[i*20])
            im = np.load(file_path).astype(np.float32)
            im = cv2.resize(im, (64, 64), interpolation=cv2.INTER_NEAREST)
            im_list.append(im.astype(np.float32))
        
        return np.array(im_list)[:,None,:,:]


def visualize_rollout(rollout, interval=50, show_step=False):
    """Visualization for a single sample rollout of a physical system.

    Args:
        rollout (numpy.ndarray): Numpy array containing the sequence of images. It's shape must be
            (seq_len, height, width, channels).
        interval (int): Delay between frames (in millisec).
        show_step (bool): Whether to draw the step number in the image
    """
    fig = plt.figure()
    img = []
    for i, im in enumerate(rollout):
        if show_step:
            black_img = np.zeros(list(im.shape))
            cv2.putText(
                black_img, str(i), (0, 30), fontScale=0.22, color=(255, 255, 255), thickness=1,
                fontFace=cv2.LINE_AA)
            res_img = (im + black_img / 255.) / 2
        else:
            res_img = im
        img.append([plt.imshow(res_img, animated=True)])
    ani = animation.ArtistAnimation(fig,
                                    img,
                                    interval=interval,
                                    blit=True,
                                    repeat_delay=100)
    plt.show()


if __name__=="__main__":
    ds = DroplerLoader('../trajectories')
    train_ds = DataLoader(ds, batch_size=1, shuffle=True)
    sample = next(iter(train_ds))[0]
    visualize_rollout(sample[:,0,:,:])
    for sample in train_ds:
        print(sample.size())