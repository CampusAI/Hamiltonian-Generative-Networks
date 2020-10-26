import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.distributions as distributions
from tqdm import tqdm

from networks import NHF
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from utilities.

def plot_dataset(dataset):
    sns.kdeplot(
        dataset[0],
        dataset[1],
        bw=0.3,
        fill=True,
    )
    plt.show()

class RandomDistribution(torch.utils.data.Dataset):
    def __init__(self, points, img_file="distribution.png", shape=(32, 32)):
        img = cv2.imread("distribution.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.resize(img, shape)
        self.shape = shape

        probs = np.array(img, dtype=np.float32).flatten(order='C')
        self.probs = probs / np.sum(probs)
        self.indexes = np.array(list(range(probs.size)))
        
        self.dataset_length = points
        # self.dataset = np.swapaxes(self.sample(points), 0, 1)
        self.dataset = np.random.multivariate_normal(np.zeros(2), np.eye(2), size=points)
        self.dataset = np.array(self.dataset, dtype=np.float32)

    def sample(self, points):
        sample = np.random.choice(self.indexes, p=self.probs, size=points)
        coord = np.unravel_index(sample, shape=self.shape, order='C')
        return coord

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, i):
        return self.dataset[i]


if __name__ == "__main__":
    final_distribution = RandomDistribution(points=1000)
    # plot_dataset(np.swapaxes(np.array(final_distribution.dataset), 0, 1))
    train = torch.utils.data.DataLoader(final_distribution, batch_size=10, shuffle=True)
    
    src_distribution = distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.eye(2))
    nhf = NHF(input_size=2, delta_t=0.1, flow_steps=1, src_distribution=src_distribution)

    optim_params = [{'params': flow.parameters()} for flow in nhf.flows]
    optim_params.append({'params': nhf.encoder.parameters()})
    optimizer = torch.optim.Adam(optim_params, lr=3.0e-3)
    
    lagrange_multiplier = 1.0
    epochs = 20
    errors = []
    for epoch in range(epochs): 
        for batch in tqdm(train):
            optimizer.zero_grad()
            batch.requires_grad_(True)
            neg_elbo = -nhf.elbo(q=batch, lagrange_multiplier=lagrange_multiplier)
            neg_elbo.backward()
            optimizer.step()
        print(neg_elbo)
        errors.append(neg_elbo)

    plt.plot(errors)
    plt.show()    

    samples = []
    for _ in range(100):
        sample = nhf.sample().detach().cpu().numpy()
        samples.append(sample)
    samples = np.squeeze(np.array(samples), axis=1)
    print(samples.shape)
    plot_dataset(np.swapaxes(np.array(samples), 0, 1))

    # plot_dataset(distro.sample(1000))