import os
import torch
import torchvision
import numpy as np
from torch import nn
from pytorch_lightning import LightningModule
from modules import Generator, Discriminator, Classifier, DiceLoss
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.trainer import Trainer
from utils import make_scatter, draw_auc_chart, draw_snr_dice_chart
from datasets import preprocess_data, collate_fn_tuples
import tensorboard

class WGANGP(LightningModule):
	def __init__(self,
				lr: float = 0.0001,
				b1: float = 0.5,
				b2: float = 0.9,
				lambda_gp: float = 10.,
				generator_loss_ratio: float = 0.75,
				batch_size: int = 32, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.lr = lr
		self.b1 = b1
		self.b2 = b2
		self.batch_size = batch_size
		self.lambda_gp = lambda_gp
		self.generator_loss_ratio = generator_loss_ratio

		# modules
		self.generator = Generator()
		self.discriminator = Discriminator()
		self.classifier = Classifier()
		self.dice_loss = DiceLoss()
		self.bce_wl = nn.BCEWithLogitsLoss()

	def forward(self, z):
		return self.generator(z)

	def compute_gradient_penalty(self, real_samples, fake_samples):
		"""Calculates the gradient penalty loss for WGAN GP"""
		alpha = torch.rand(real_samples.size(0), 1, 1, dtype=real_samples.dtype, device=self.device)
		# Get random interpolation between real and fake samples
		interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
		interpolates = interpolates.to(self.device)
		d_interpolates = self.discriminator(interpolates)
		fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
		# Get gradient w.r.t. interpolates
		gradients = torch.autograd.grad(
			outputs=d_interpolates,
			inputs=interpolates,
			grad_outputs=fake,
			create_graph=True,
			retain_graph=True,
			only_inputs=True,
		)[0]
		gradients = gradients.view(gradients.size(0), -1).to(self.device)
		gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
		return gradient_penalty

	def training_step(self, batch, batch_idx, optimizer_idx):
		lightcurves, targets, class_lightcurves = batch
		lightcurves = self.pad(lightcurves)
		targets = (self.pad(targets) > 0).float()
		class_lightcurves = self.pad(class_lightcurves)

		# train generator
		if optimizer_idx == 0:
			output, _ = self.generator(lightcurves)

			if self.global_step % 100 == 0:
				lc_out = output[0].clone().detach().cpu().numpy()
				lc_in = lightcurves[0].clone().detach().cpu().numpy()
				tar_in = targets[0].clone().detach().cpu().numpy()
				fig = make_scatter(lc_in[0], lc_out[0], tar_in[0], self.dice_loss(output[0], targets[0]).detach().cpu().numpy())
				self.logger.experiment.add_figure('sample output', fig, self.global_step)

			g_loss = -torch.mean(self.discriminator(output)) * (1.-self.generator_loss_ratio)
			dice_loss = torch.mean(self.dice_loss(output, targets)) * self.generator_loss_ratio
			self.log("dice_loss", dice_loss, prog_bar=True)
			self.log("g_loss", g_loss, prog_bar=True)
			output = OrderedDict({
				'loss': g_loss + dice_loss,
			})
			return output

		# train discriminator
		elif optimizer_idx == 1:
			gen_output, _ = self.generator(lightcurves)

			# Real images
			real_validity = self.discriminator(targets)
			# Fake images
			fake_validity = self.discriminator(gen_output)
			# Gradient penalty
			gradient_penalty = self.compute_gradient_penalty(targets.data, gen_output.data)
			# Adversarial loss
			d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty

			self.log("d_loss", d_loss, prog_bar=True)
			output = OrderedDict({
				'loss': d_loss,
			})
			return output

		elif optimizer_idx == 2:
			in_lightcurves = torch.cat([lightcurves[:lightcurves.size(0)//2].clone(), class_lightcurves[:class_lightcurves.size(0)//2].clone()])
			class_target = torch.cat([torch.ones(lightcurves.size(0)//2, 1), torch.zeros(class_lightcurves.size(0)//2, 1)])

			gen_outputs, classifier_inputs = self.generator(in_lightcurves)
			classifier_logits = self.classifier(gen_outputs, classifier_inputs)
			c_loss = self.bce_wl(classifier_logits.cpu(), class_target.cpu())
			self.log("c_loss", c_loss, prog_bar=True)
			output = OrderedDict({
				'loss': c_loss,
			})
			return output

	def validation_step(self, batch, batch_idx, *args, **kwargs):
		lightcurves, targets, class_lightcurves, snrs = batch
		lightcurves = self.pad(lightcurves)
		targets = (self.pad(targets) > 0).float()
		class_lightcurves = self.pad(class_lightcurves)

		gen_output, _ = self.generator(lightcurves)
		dice_results = 1 - self.dice_loss(gen_output, targets)
		in_lightcurves = torch.cat([lightcurves, class_lightcurves])
		class_target = torch.cat([torch.ones(lightcurves.size(0), 1), torch.zeros(class_lightcurves.size(0), 1)])
		gen_outputs, classifier_inputs = self.generator(in_lightcurves)
		classifier_logits = self.classifier(gen_outputs, classifier_inputs)


		out = [dice_results, snrs, torch.sigmoid(classifier_logits), class_target, lightcurves]
		return out
		
	def validation_epoch_end(self, outputs, step=-1):
		dice, snrs = torch.concat([o[0] for o in outputs], dim=0).cpu(), torch.concat([o[1] for o in outputs], dim=0).cpu()
		y_pred, y_true = torch.concat([o[2] for o in outputs], dim=0).cpu().squeeze(1), torch.concat([o[3] for o in outputs], dim=0).cpu().squeeze(1)
		self.logger.experiment.add_figure('AUC Chart', draw_auc_chart(y_pred, y_true), self.global_step)
		self.logger.experiment.add_figure('Dice-SNR Chart', draw_snr_dice_chart(dice.numpy(), snrs.numpy()), self.global_step)
		self.logger.experiment.add_figure('Dice-SNR Zoom Chart', draw_snr_dice_chart(dice.numpy(), snrs.numpy(), True), self.global_step)
		
	def test_step(self, batch, batch_idx, *args, **kwargs):
		return self.validation_step(batch, batch_idx, *args, **kwargs)

	def test_epoch_end(self, outputs):
		dice, snrs = torch.concat([o[0] for o in outputs], dim=0).cpu(), torch.concat([o[1] for o in outputs], dim=0).cpu()
		y_pred, y_true = torch.concat([o[2] for o in outputs], dim=0).cpu().squeeze(1), torch.concat([o[3] for o in outputs], dim=0).cpu().squeeze(1)
		x = torch.concat([o[4] for o in outputs], dim=0).cpu().squeeze(1)
		self.logger.experiment.add_figure('Test AUC Chart', draw_auc_chart(y_pred, y_true), self.global_step+1)
		self.logger.experiment.add_figure('Test Dice-SNR Chart', draw_snr_dice_chart(dice.numpy(), snrs.numpy()), self.global_step+1)
		self.logger.experiment.add_figure('Test Dice-SNR Zoom Chart', draw_snr_dice_chart(dice.numpy(), snrs.numpy(), True), self.global_step+1)

		os.makedirs('./outputs', exist_ok=True)
		np.save('./outputs/output_y_pred.npy', y_pred.numpy())
		np.save('./outputs/output_y_true.npy', y_true.numpy())
		np.save('./outputs/output_x.npy', x.numpy())
		draw_auc_chart(y_pred.numpy(), y_true.numpy()).savefig('./outputs/test_auc_chart.png')
		draw_snr_dice_chart(dice.numpy(), snrs.numpy()).savefig('./outputs/test_dice_snr.png')
		draw_snr_dice_chart(dice.numpy(), snrs.numpy(), True).savefig('./outputs/test_zoom_dice_snr.png')

	def configure_optimizers(self):
		n_generator = 1
		n_critic = 1
		n_classifier = 1

		lr = self.lr
		b1 = self.b1
		b2 = self.b2

		opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
		opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
		opt_c = torch.optim.Adam(self.classifier.parameters(), lr=lr * 10, betas=(0.9, 0.99))
		return (
			{'optimizer': opt_g, 'frequency': n_generator},
			{'optimizer': opt_d, 'frequency': n_critic},
			{'optimizer': opt_c, 'frequency': n_classifier}
		)
	def train_dataloader(self):
		tensor_o = torch.Tensor(np.load('../data/processed/total_original_sim_train.npy'))
		tensor_x = torch.Tensor(np.load('../data/processed/total_added_t_sim_train.npy'))
		tensor_y = torch.Tensor(np.load('../data/processed/total_transits_sim_train.npy'))
		
		padding = int(np.ceil(tensor_x.size(-1) / 162.)) * 162 - tensor_x.size(-1)
		self.pad = torchvision.transforms.Pad([0, 0, padding, 0])

		t_dataset = TensorDataset(tensor_x, tensor_y, tensor_o)
		return DataLoader(t_dataset, collate_fn=collate_fn_tuples, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=8)

	def val_dataloader(self):
		tensor_o = torch.Tensor(np.load('../data/processed/total_original_sim_val.npy'))
		tensor_x = torch.Tensor(np.load('../data/processed/total_added_t_sim_val.npy'))
		tensor_y = torch.Tensor(np.load('../data/processed/total_transits_sim_val.npy'))
		tensor_snr = torch.Tensor(np.load('../data/processed/total_params_sim_val.npy'))[:, -1]
		
		padding = int(np.ceil(tensor_x.size(-1) / 162.)) * 162 - tensor_x.size(-1)
		self.pad = torchvision.transforms.Pad([0, 0, padding, 0])

		t_dataset = TensorDataset(tensor_x, tensor_y, tensor_o, tensor_snr)
		return DataLoader(t_dataset, collate_fn=collate_fn_tuples, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=8)

	def test_dataloader(self):
		tensor_o = torch.Tensor(np.load('../data/processed/total_original_sim_test.npy'))
		tensor_x = torch.Tensor(np.load('../data/processed/total_added_t_sim_test.npy'))
		tensor_y = torch.Tensor(np.load('../data/processed/total_transits_sim_test.npy'))
		tensor_snr = torch.Tensor(np.load('../data/processed/total_params_sim_test.npy'))[:, -1]

		
		padding = int(np.ceil(tensor_x[0].size(-1) / 162.)) * 162 - tensor_x[0].size(-1)
		self.pad = torchvision.transforms.Pad([0, 0, padding, 0])

		t_dataset = TensorDataset(tensor_x, tensor_y, tensor_o, tensor_snr)
		return DataLoader(t_dataset, collate_fn=collate_fn_tuples, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=8)


def main(args) -> None:
	# ------------------------
	# 1 INIT LIGHTNING MODEL
	# ------------------------
	preprocess_data()

	model = WGANGP()

	# ------------------------
	# 2 INIT TRAINER
	# ------------------------
	trainer = Trainer(accelerator='gpu', devices=args.gpus, max_epochs=96)

	# ------------------------
	# 3 START TRAINING
	# ------------------------
	trainer.fit(model)
	trainer.test(model)


if __name__ == '__main__':
	from argparse import ArgumentParser
	parser = ArgumentParser()
	parser.add_argument("--gpus", type=int, default=0, help="number of GPUs")
	hparams = parser.parse_args()

	main(hparams)