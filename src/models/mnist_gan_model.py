from typing import Union, Dict, Any, Tuple, Optional

import wandb
import torch
import torchvision
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import LightningModule


class MNISTGANModel(LightningModule):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = generator
        self.discriminator = discriminator
        self.adversarial_loss = torch.nn.MSELoss()

    def forward(self, z, labels) -> Tensor:
        return self.generator(z, labels)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2)
        )
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx, optimizer_idx) -> Union[Tensor, Dict[str, Any]]:
        log_dict, loss = self.step(batch, batch_idx, optimizer_idx)
        self.log_dict({"/".join(("train", k)): v for k, v in log_dict.items()})
        return loss

    def validation_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        log_dict, loss = self.step(batch, batch_idx)
        self.log_dict({"/".join(("val", k)): v for k, v in log_dict.items()})
        return None

    def test_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        # TODO: if you have time, try implementing a test step
        raise NotImplementedError

    def step(self, batch, batch_idx, optimizer_idx=None):
        imgs, labels = batch
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = torch.ones((batch_size, 1), device=self.device)
        fake = torch.zeros((batch_size, 1), device=self.device)

        # Configure input
        z = torch.randn((batch_size, self.hparams.latent_dim), device=self.device)

        log_dict = {}
        loss = None

        if optimizer_idx == 0 or not self.training:
            # Generate a batch of images
            gen_imgs = self(z, labels)

            # Loss measures generator's ability to fool the discriminator
            g_loss = self.adversarial_loss(self.discriminator(gen_imgs, labels), valid)

            log_dict["g_loss"] = g_loss
            loss = g_loss

        if optimizer_idx == 1 or not self.training:
            # Measure discriminator's ability to classify real from generated samples
            gen_imgs = self(z, labels)
            
            real_loss = self.adversarial_loss(self.discriminator(imgs, labels), valid)
            fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach(), labels), fake)
            d_loss = (real_loss + fake_loss) / 2

            log_dict["d_loss"] = d_loss
            loss = d_loss

        return log_dict, loss

    def on_epoch_end(self):
        z = torch.randn(8, self.hparams.latent_dim, device=self.device)
        labels = torch.randint(0, self.hparams.n_classes, (8,), device=self.device)
        sample_imgs = self(z, labels)
        grid = torchvision.utils.make_grid(sample_imgs, nrow=4, normalize=True)

        for logger in self.trainer.logger:
            if type(logger).__name__ == "WandbLogger":
                logger.experiment.log({"gen_imgs": [wandb.Image(grid)]})
