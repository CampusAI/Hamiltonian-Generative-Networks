from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    def __init__(self, loss_freq=100, rollout_freq=1000):
        self.writer = SummaryWriter()
        self.iteration = 0
        self.loss_freq = loss_freq
        self.rollout_freq = rollout_freq

    def step(self, losses, rollout_batch, prediction):
        if self.iteration % self.loss_freq == 0:
            self.writer.add_scalar('data/reconstruction_loss', losses[0],
                                   self.iteration)
            self.writer.add_scalar('data/kld_loss', losses[1], self.iteration)

        if self.iteration % self.rollout_freq == 0:
            self.writer.add_video('data/input',
                                  rollout_batch.detach().cpu(), self.iteration)
            self.writer.add_video(
                'data/reconstruction',
                prediction.reconstructed_rollout.unsqueeze(2).detach().cpu(),
                self.iteration)

        self.iteration += 1