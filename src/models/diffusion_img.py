import math
from pathlib import Path
import glob
from dataclasses import asdict
import logging

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from accelerate import Accelerator
import wandb

from diffusers import DDIMPipeline, DDIMScheduler
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid

from src.utils.datasets import get_img_dataset


logger = logging.getLogger(__name__)


def evaluate(config, epoch, pipeline):
    """ Evaluate the model by sampling images and saving a grid. """
    # Sample some images from random noise.
    images = pipeline(
        batch_size=config.sampling.gen_batch_size,
        generator=torch.manual_seed(config.common.seed + epoch),
        num_inference_steps=config.sampling.num_inference_steps,
        output_type="pil",
    ).images

    # Save image grid
    sample_dir = Path(config.paths.image_dir) / "training_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    grid_path = sample_dir / f"{epoch:04d}.png"
    
    rows = int(math.ceil(np.sqrt(len(images))))
    cols = int(math.ceil(len(images) / rows))
    image_grid = make_image_grid(images, rows=rows, cols=cols)
    image_grid.save(grid_path)
    
    return grid_path


class DiffuserTrainer:
    """ Trainer class using diffusers. """
    def __init__(self, config, model, noise_scheduler):
        self.config = config
        self.model = model
        self.noise_scheduler = noise_scheduler

        self.run_dir = Path(self.config.paths.run_dir)
        self.log_dir = self.run_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading dataset for diffusion model training...")
        data_dict = get_img_dataset(config=config, splits=['train'])
        self.train_dataloader = data_dict.get('train_loader')
        if self.train_dataloader is None:
            raise ValueError("'train_loader' not found in dataset dictionary.")

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)

        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / config.training.gradient_accumulation_steps)
        max_train_steps = config.training.num_epochs * num_update_steps_per_epoch

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=config.training.lr_warmup_steps * config.training.gradient_accumulation_steps,
            num_training_steps=max_train_steps,
        )

        self.accelerator = Accelerator(
            mixed_precision=self.config.training.mixed_precision,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            log_with=["tensorboard", "wandb"] if self.config.common.push_to_wandb else "tensorboard",
            project_dir=str(self.log_dir),
        )

        # Initialize trackers
        if self.accelerator.is_main_process:
            if self.config.common.push_to_wandb:
                self._setup_wandb()
            else:
                self.accelerator.init_trackers(
                    project_name=self.config.common.exp_name or "ddim-diffusion-training",
                    config=asdict(self.config),
                )

    def _setup_wandb(self):
        """ Set up WandB integration. """
        run_name = self.config.common.exp_name or f"{self.config.dataset.name}-ddim-training"
        self.accelerator.init_trackers(
            project_name="diffusion_training",
            config=asdict(self.config),
            init_kwargs={"wandb": {
                "name": run_name,
                "dir": str(self.run_dir),
                "resume": "allow",
                "id": wandb.util.generate_id()
            }}
        )
        self.accelerator.print("WandB initialized.")

    def _save_checkpoint(self, epoch: int):
        """ Save training checkpoint. """
        if not self.accelerator.is_main_process:
            return

        save_path= Path(self.config.paths.diffusion_load_path)
        optimizer_save_file = save_path.parent / f"{save_path.name}_optimizer_epoch{epoch}.pth"

        if not save_path.is_dir():
            save_path.mkdir(parents=True, exist_ok=True)
            self.accelerator.print(f"Created directory for saving checkpoint: {save_path}")

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        pipeline = DDIMPipeline(unet=unwrapped_model, scheduler=self.noise_scheduler)
        pipeline.save_pretrained(save_path)

        checkpoint_state = {
            "epoch": epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
        }
        torch.save(checkpoint_state, optimizer_save_file)
        self.accelerator.print(f"Pipeline saved to {save_path}, optimizer state saved to {optimizer_save_file}")

    def _load_checkpoint(self):
        """ Load training checkpoint optimizer/scheduler state. """
        start_epoch = 0
        pipeline_load_path = Path(self.config.paths.diffusion_load_path)

        if not Path(pipeline_load_path).is_dir():
            self.accelerator.print("No valid diffusion load path provided. Starting from epoch 0.")
            return start_epoch

        optimizer_pattern = pipeline_load_path.parent / f"{pipeline_load_path.name}_optimizer_epoch*.pth"

        optimizer_checkpoints = sorted(glob.glob(str(optimizer_pattern)), reverse=True)
        if optimizer_checkpoints:
            latest_optimizer_file = optimizer_checkpoints[0]
            self.accelerator.print(f"Found optimizer checkpoint: {latest_optimizer_file}")
            checkpoint_state = torch.load(latest_optimizer_file, map_location=self.accelerator.device)

            self.optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
            self.lr_scheduler.load_state_dict(checkpoint_state["lr_scheduler_state_dict"])
            start_epoch = checkpoint_state.get("epoch", -1) + 1
            self.accelerator.print(f"Resumed optimizer/scheduler state from epoch {start_epoch-1}")
        else:
            self.accelerator.print(f"No optimizer checkpoint found for pattern: {optimizer_pattern}. Starting from epoch 0.")

        return start_epoch

    def train(self):
        """ Main training loop. """
        start_epoch = self._load_checkpoint()

        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.config.training.gradient_accumulation_steps)
        global_step = start_epoch * num_update_steps_per_epoch

        logger.info("***** Training Diffusion Model *****")
        logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        logger.info(f"  Num Epochs = {self.config.training.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.config.dataset.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.config.dataset.batch_size * self.accelerator.num_processes * self.config.training.gradient_accumulation_steps}")
        logger.info(f"  Gradient Accumulation steps = {self.config.training.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.config.training.num_epochs * num_update_steps_per_epoch}")
        logger.info(f"  Starting epoch = {start_epoch}")
        logger.info(f"  Starting step = {global_step}")

        for epoch in range(start_epoch, self.config.training.num_epochs):
            self.model.train()
            progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            epoch_loss = 0.0

            for step, batch in enumerate(self.train_dataloader):
                clean_images = batch[self.config.dataset.x_tensor]
                noise = torch.randn(clean_images.shape, device=clean_images.device)

                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],),
                    device=clean_images.device, dtype=torch.int64
                )
                noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

                with self.accelerator.accumulate(self.model):
                    noise_pred = self.model(noisy_images, timesteps, return_dict=False)[0]
                    loss = F.mse_loss(noise_pred, noise)

                    avg_loss = self.accelerator.gather(loss.repeat(self.config.dataset.batch_size)).mean()
                    epoch_loss += avg_loss.item() / self.config.training.gradient_accumulation_steps

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    logs = {"loss": avg_loss.item(), "lr": self.lr_scheduler.get_last_lr()[0], "epoch": epoch, "step": global_step}
                    progress_bar.set_postfix(**logs)
                    self.accelerator.log(logs, step=global_step)

            progress_bar.close()
            if self.accelerator.is_main_process:
                avg_epoch_loss = epoch_loss / len(self.train_dataloader)
                self.accelerator.log({"epoch_loss": avg_epoch_loss}, step=global_step)

                pipeline = None
                if (epoch % self.config.training.save_image_epochs == 0) or (epoch == self.config.training.num_epochs - 1):
                    with torch.no_grad():
                        pipeline = DDIMPipeline(unet=self.accelerator.unwrap_model(self.model),
                                                scheduler=DDIMScheduler.from_config(self.noise_scheduler.config))
                        img_filename = evaluate(self.config, epoch, pipeline)
                        if img_filename and self.config.common.push_to_wandb:
                            wandb.log({"samples": wandb.Image(img_filename, caption=f"Epoch: {epoch}")}, step=global_step)

                if (epoch % self.config.training.save_model_epochs == 0) or (epoch == self.config.training.num_epochs - 1):
                    self._save_checkpoint(epoch)

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
        logger.info('Training complete')


def get_diffusion_model(config):
    """
    Retrieves or trains a diffusion model pipeline (DDIM).
    For MNIST, the model is trained from scratch.
    For CIFAR10, the model is loaded from google/ddpm-cifar10-32.
    """
    # Load pipeline from HF for sanity check
    if config.dataset.name == 'cifar10':
        pipeline = DDIMPipeline.from_pretrained(config.paths.diffusion_load_path)
        return pipeline

    model = UNet2DModel(
        sample_size=config.dataset.image_size,
        in_channels=config.dataset.channels,
        out_channels=config.dataset.channels,
        layers_per_block=2,
        block_out_channels=config.dataset.block_out_channels,
    )

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type='epsilon',
        clip_sample=False,
    )

    load_path = config.paths.diffusion_load_path
    pipeline = None
    do_training = False

    # Load pretrained model from local else train new model
    if not load_path:
        ckpt_path = Path(config.paths.checkpoint_dir) / f"{config.dataset.name}_unet_{config.dataset.image_size}px"
        logger.info(f"No valid diffusion load path provided. Saving to: {ckpt_path}")
        do_training = True
    else:
        ckpt_path = Path(load_path)
        logger.info(f"Loading existing diffusion pipeline from: {ckpt_path}")
        try:
            pipeline = DDIMPipeline.from_pretrained(ckpt_path)
            logger.info(f"Successfully loaded diffusion pipeline from {ckpt_path}")
            model = pipeline.unet
            noise_scheduler = pipeline.scheduler
        except Exception as e:
            logger.warning(f"Failed to load diffusion pipeline from '{ckpt_path}'. Training new model. Error: {e}.")
            do_training = True

    if do_training:
        config.paths.diffusion_load_path = ckpt_path
        ckpt_path.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing DiffuserTrainer...")
        trainer = DiffuserTrainer(
            config=config,
            model=model,
            noise_scheduler=noise_scheduler,
        )

        logger.info("Starting diffusion model training...")
        trainer.train()

        logger.info(f"Training complete. Diffusion pipeline saved to {ckpt_path}")
        pipeline = DDIMPipeline.from_pretrained(ckpt_path)

    if pipeline is None:
        logger.error("Diffusion pipeline is None after load/train attempt. Cannot proceed.")
        raise RuntimeError("Failed to obtain diffusion model pipeline.")

    logger.info(f'Diffusion UNet2DModel - Total params: {(sum(p.numel() for p in model.parameters()) / 1e6):.2f}M ')
    return pipeline
