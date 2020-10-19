import os

from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:

    def __init__(self, hyper_params, loss_freq=100, rollout_freq=1000):
        """Instantiate a TrainingLogger.

        Args:
            hyper_params (dict): Parameters used to train the model (for reproducibility).
            loss_freq (int, optional): Frequency at which the loss values are updated in
                TensorBoard. Defaults to 100.
            rollout_freq (int, optional): Frequency at which videos are updated in TensorBoard.
                Defaults to 1000.
        """
        self.writer = SummaryWriter(log_dir=os.path.join("runs", hyper_params["experiment_id"]))
        self.writer.add_text('data/hyperparams', str(hyper_params), 0)
        self.iteration = 0
        self.loss_freq = loss_freq
        self.rollout_freq = rollout_freq

    def step(self, losses, rollout_batch, prediction):
        """Perform a logging step: Update inner iteration counter and log info if needed

        Args:
            losses (tuple): Tuple of two floats, corresponding to reconstruction loss and KLD.
            rollout_batch (torch.Tensor): Batch of ground truth rollouts, as a Tensor of shape
                (batch_size, seq_len, channels, height, width).
            prediction (utilities.hgn_result.HgnResult): The HgnResult object containing data of
                the models forward pass on the rollout_batch.
        """
        if self.iteration % self.loss_freq == 0:
            self.writer.add_scalar('data/reconstruction_loss', losses[0],
                                   self.iteration)
            self.writer.add_scalar('data/kld_loss', losses[1], self.iteration)

        if self.iteration % self.rollout_freq == 0:
            self.writer.add_video('data/input',
                                  rollout_batch.detach().cpu(), self.iteration)
            self.writer.add_video(
                'data/reconstruction',
                prediction.reconstructed_rollout.detach().cpu(),
                self.iteration)
        self.iteration += 1
