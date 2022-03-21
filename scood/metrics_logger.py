from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ImportError:
    pass


class MetricsLogger:
    def __init__(self, log_dir, use_wandb=False, **kwargs):
        if use_wandb:

            wandb.tensorboard.patch(
                root_logdir=str(log_dir),
            )
            wandb.init(
                name=kwargs["name"],
                project=kwargs["project"],
                config=kwargs["config"],
                id=kwargs["id"],
                group=kwargs['group'],
                resume="allow",
            )

        self.writer = SummaryWriter(log_dir)

    def write_scalar(self, tag, scalar_value, global_step=None):
        self.writer.add_scalar(tag, scalar_value, global_step)

    def write_scalar_dict(self, scalar_dict, global_step=None):
        for tag, scalar_value in scalar_dict.items():
            self.write_scalar(tag, scalar_value, global_step)

    def write_histogram(self):
        pass

    def write_images(self):
        pass

    def write_figure(self):
        pass

    def write_embedding(self):
        pass

    def write_hparams(self):
        pass

    def close(self):
        self.writer.close()
