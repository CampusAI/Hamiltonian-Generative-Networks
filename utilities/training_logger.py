import os

from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    def __init__(self,
                 hyper_params,
                 loss_freq=100,
                 rollout_freq=1000,
                 model_freq=1000):
        """Instantiate a TrainingLogger.

        Args:
            hyper_params (dict): Parameters used to train the model (for reproducibility).
            loss_freq (int, optional): Frequency at which the loss values are updated in
                TensorBoard. Defaults to 100.
            rollout_freq (int, optional): Frequency at which videos are updated in TensorBoard.
                Defaults to 1000.
            model_freq (int, optional): Frequency at which a checkpoint of the model is saved.
                Defaults to 1000.
        """
        self.writer = SummaryWriter(
            log_dir=os.path.join("runs", hyper_params["experiment_id"]))
        self.writer.add_text('data/hyperparams', str(hyper_params), 0)
        self.hparams = hyper_params
        self.iteration = 0
        self.loss_freq = loss_freq
        self.rollout_freq = rollout_freq
        self.model_freq = model_freq

    def step(self, losses, rollout_batch, prediction, model):
        """Perform a logging step: update inner iteration counter and log info if needed.

        Args:
            losses (tuple): Tuple of two floats, corresponding to reconstruction loss and KLD.
            rollout_batch (torch.Tensor): Batch of ground truth rollouts, as a Tensor of shape
                (batch_size, seq_len, channels, height, width).
            prediction (utilities.hgn_result.HgnResult): The HgnResult object containing data of
                the models forward pass on the rollout_batch.
        """
        if self.iteration % self.loss_freq == 0:
            for loss_name, loss_value in losses.items():
                if loss_value is not None:
                    self.writer.add_scalar(f'{loss_name}', loss_value, self.iteration)
            enery_mean, energy_std = prediction.get_energy()
            self.writer.add_scalar(f'energy/mean', enery_mean, self.iteration)
            self.writer.add_scalar(f'energy/std', energy_std, self.iteration)

        if self.iteration % self.rollout_freq == 0:
            self.writer.add_video('data/input',
                                  rollout_batch.detach().cpu(), self.iteration)
            self.writer.add_video(
                'data/reconstruction',
                prediction.reconstructed_rollout.detach().cpu(),
                self.iteration)

            # Sample from HGN and add to tensorboard
            random_sample = model.get_random_sample(n_steps=50, img_shape=(32, 32))
            self.writer.add_video(
                'data/sample',
                random_sample.reconstructed_rollout.detach().cpu(),
                self.iteration)
    
        if self.iteration % self.model_freq == 0:
            save_dir = os.path.join(
                self.hparams["model_save_dir"], self.hparams["experiment_id"] +
                "_checkpoint_" + str(self.iteration))
            model.save(save_dir)
        self.iteration += 1

    def log_text(self, label, msg):
        """Add text to tensorboard
        Args:
            label (str): Label to identify in tensorboard display
            msg (str, float): Message to display (can be a numericsl value)
        """
        self.writer.add_text('data/' + label, str(msg), 0)
        
    def log_error(self, label, mean, dist):
        """Add text to tensorboard
        Args:
            mean (float): Mean of the error interval to display.
            dist (float): distance of the error corresponding to the confidence.
        """
        self.log_text(label, "{:.8f} +/- {:.8f}".format(mean, dist))
