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
from tqdm import tqdm
from torch.nn import functional as F


class DroplerLoader(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)
        print("Preprocessing data...")
        self.tensor_list = []
        for i in tqdm(range(len(self.file_list))):
            path = os.path.join(self.root_dir, self.file_list[i])
            image_list = sorted(os.listdir(path))
            im_list = []
            seq_len = len(image_list)
            for j in range(len(image_list)):
                file_path = os.path.join(path, image_list[j])
                im = np.load(file_path).astype(np.float32)
                im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
                im_list.append(im)
            im = np.array(im_list)
            mod = seq_len%20
            im = im[:seq_len-mod,:,:]
            im = im.reshape(len(im)//20,20, 256, 256).transpose((1,0,2,3))[:,:,None,:,:]
            self.tensor_list.append(im)
        self.tensor_list
        print("Preprocessed data! Length is", len(self.tensor_list))


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        """
        path = os.path.join(self.root_dir, self.file_list[i])
        im = np.load(path).astype(np.float32)
        seq_len, h, w = im.shape
        mod = seq_len%20
        im = im[:seq_len-mod,:,:]
        return im.reshape(len(im)//20,20, h, w).transpose((1,0,2,3))[:,:,None,:,:]
        """
        return self.tensor_list[i]

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

def calculate_mse_lower_bound(root_dir):

    file_list = os.listdir(root_dir)
    sums_list = []
    pixel_sum = []
    for i in tqdm(range(len(file_list))):
        path = os.path.join(root_dir, file_list[i])
        image_list = sorted(os.listdir(path))
        mse_list = []
        seq_len = len(image_list)
        pixel_avg = []
        for j in range(len(image_list)):
            file_path = os.path.join(path, image_list[j])
            im_orig = np.load(file_path).astype(np.float32)
            
            im = cv2.resize(im_orig, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
            im = np.round(cv2.resize(im, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32))
            pixel_avg.append(im_orig.sum())
            mse_list.append(F.mse_loss(torch.from_numpy(im_orig), torch.from_numpy(im), reduction='sum'))
        sums = np.array(mse_list)
        s = sums.sum()/(seq_len*1024*1024)
        sums_list.append(s)
        pixel_sum.append(np.array(pixel_avg).sum()/(seq_len*1024*1024))

    result = np.array(sums_list).mean()
    avg_pixel_val = np.array(pixel_sum).mean()
    print(result)
    print(avg_pixel_val)
        

if __name__=="__main__":
    calculate_mse_lower_bound('../trajectories')
    #ds = DroplerLoader('../trajectories')
    #train_ds = DataLoader(ds, batch_size=None, shuffle=True)
    #sample = next(iter(train_ds))[0]
    #visualize_rollout(sample[:,0,:,:])
    #for sample in train_ds:
    #    print(sample.size())