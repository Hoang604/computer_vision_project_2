import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter # Use PyTorch's TensorBoard
import numpy as np
from tqdm import tqdm
import os
import bitsandbytes as bnb
import datetime
from diffusers import DDIMScheduler # Keep this import
from torch.utils.data import DataLoader
from unet import UNet

# --- Diffusion Model ---
class DiffusionModel:
    """
    Implements a Denoising Diffusion Probabilistic Model (DDPM).

    This class handles the forward (noising) and reverse (denoising)
    processes, as well as training and sampling logic. It uses a cosine
    noise schedule by default. The model can be trained to predict either
    the noise added during the forward process or a 'v-prediction' target.

    Attributes:
        timesteps (int): The total number of timesteps in the diffusion process.
        device (str or torch.device): The device on which to perform computations ('cuda' or 'cpu').
        mode (str): The prediction mode, either "v_prediction" or "noise".
        betas (torch.Tensor): Noise schedule (variance of noise added at each step).
        alphas (torch.Tensor): 1.0 - betas.
        alphas_cumprod (torch.Tensor): Cumulative product of alphas.
        alphas_cumprod_shift_right (torch.Tensor): Cumulative product of alphas, shifted right by one.
        sqrt_alphas_cumprod (torch.Tensor): Square root of alphas_cumprod.
        sqrt_one_minus_alphas_cumprod (torch.Tensor): Square root of (1.0 - alphas_cumprod).
        sqrt_recip_alphas (torch.Tensor): Square root of the reciprocal of alphas.
        posterior_variance (torch.Tensor): Variance of the posterior distribution q(x_{t-1} | x_t, x_0).
    """
    def __init__(
        self,
        timesteps=1000,
        device='cuda',
        mode="v_prediction" # Added mode attribute
    ):
        """
        Initializes the DiffusionModel with a specified number of timesteps, device, and prediction mode.

        It pre-computes various parameters of the diffusion process based on a
        cosine noise schedule.

        Args:
            timesteps (int, optional): Number of diffusion steps. Defaults to 1000.
            device (str or torch.device, optional): Device for tensor operations ('cuda' or 'cpu').
                                                   Defaults to 'cuda'.
            mode (str, optional): The prediction mode for the model during training.
                                  Can be "v_prediction" or "noise".
                                  Defaults to "v_prediction".
        Raises:
            ValueError: If the provided `mode` is not "v_prediction" or "noise".
        """
        self.timesteps = timesteps
        self.device = device
        
        if mode not in ["v_prediction", "noise"]:
            raise ValueError("Mode must be 'v_prediction' or 'noise'")
        self.mode = mode
        print(f"DiffusionModel initialized in '{self.mode}' mode.")


        # Define cosine noise schedule using PyTorch tensors
        def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
            """
            Generates a cosine noise schedule (betas).

            Args:
                timesteps (int): The number of timesteps.
                s (float, optional): Small offset to prevent beta_t from being too small near t=0.
                                     Defaults to 0.008.
                dtype (torch.dtype, optional): Data type for the tensors. Defaults to torch.float32.

            Returns:
                torch.Tensor: The beta schedule.
            """
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=dtype)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0., 0.999) # Use PyTorch's clip

        self.betas = cosine_beta_schedule(timesteps).to(self.device)

        # Pre-calculate diffusion parameters using PyTorch tensors
        self.alphas = (1.0 - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        # Use torch.cat instead of np.append for PyTorch tensors
        self.alphas_cumprod_shift_right = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(self.device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(self.device)

        # Parameters for sampling
        self.posterior_variance = (self.betas *
            (1.0 - self.alphas_cumprod_shift_right) /
            (1.0 - self.alphas_cumprod)).to(self.device)
        # Ensure no NaN from division by zero (can happen at t=0 if not careful)
        self.posterior_variance = torch.nan_to_num(self.posterior_variance, nan=0.0, posinf=0.0, neginf=0.0)


    def _extract(self, a, t, x_shape):
        """
        Helper function to extract specific coefficients at given timesteps `t` and reshape
        them to match the batch shape of `x_shape`.

        This is used to gather the appropriate alpha, beta, etc., values for a batch
        of images at different timesteps.

        Args:
            a (torch.Tensor): The tensor to extract coefficients from (e.g., alphas_cumprod).
            t (torch.Tensor): A 1D tensor of timesteps for each item in the batch. Shape [B,].
            x_shape (torch.Size): The shape of the data tensor x (e.g., [B, C, H, W]).

        Returns:
            torch.Tensor: A tensor of extracted coefficients, reshaped to [B, 1, 1, 1]
                          (or [B, 1, ..., 1] to match x_shape's dimensions) for broadcasting.
        """
        batch_size = t.shape[0]
        # Use tensor indexing instead of tf.gather
        out = a[t]
        # Use view for reshaping
        return out.view(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_0, t):
        """
        Performs the forward diffusion process (noising) q(x_t | x_0).

        It takes an initial clean image `x_0` and a timestep `t`, and returns
        a noised version `x_t` along with the noise that was added.
        The formula used is: x_t = sqrt(alphas_cumprod_t) * x_0 + sqrt(1 - alphas_cumprod_t) * noise.

        Args:
            x_0 (torch.Tensor): Input clean images, shape [B, C, H, W].
                                Assumed to be in the [-1, 1] range initially.
            t (torch.Tensor): Timesteps for each image in the batch, shape [B,].

        Returns:
            tuple:
                - torch.Tensor: Noisy images `x_t` at timestep `t`, range [-1, 1].
                - torch.Tensor: The noise `epsilon` added to the images.
        """
        x_0 = x_0.to(self.device) # Move to correct device
        x_0 = x_0.float()

        # Create random noise
        noise = torch.randn_like(x_0, device=x_0.device)

        # Get pre-calculated parameters using the helper function
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        # Forward diffusion equation: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        x_t = torch.clamp(x_t, -1.0, 1.0) # Ensure output is within valid range

        return x_t, noise

    def train(self, 
              dataset:DataLoader, 
              model:UNet, 
              optimizer:bnb.optim.Adam8bit, 
              accumulation_steps=32, epochs=30, 
              start_epoch=0, best_loss=float('inf'), 
              context="LR",
              log_dir=None, 
              checkpoint_dir=None, 
              log_dir_base="/media/hoangdv/cv_logs", 
              checkpoint_dir_base="/media/hoangdv/cv_checkpoints") -> None:
        """
        Trains the diffusion model using gradient accumulation.

        This method iterates over the dataset for a specified number of epochs,
        accumulating gradients over several batches before performing an optimizer step.
        It logs training progress to TensorBoard and saves model checkpoints.
        The model is trained to predict a target determined by `self.mode`:
        - If `self.mode` is "v_prediction", the model predicts `v`.
        - If `self.mode` is "noise", the model predicts the noise `epsilon`.

        Args:
            dataset (torch.utils.data.DataLoader): DataLoader providing training batches.
            model (torch.nn.Module): The neural network model to be trained (e.g., a U-Net).
                                     It should take `x_t` and `t` as input and predict the target
                                     corresponding to `self.mode`.
            optimizer (bnb.optim.Adam8bit): The optimizer (e.g., Adam8bit from bitsandbytes).
            image_generator (ResidualGenerator): An instance of ResidualGenerator for sampling images
                                                 during training (not currently used in this loop
                                                 but passed as an argument).
            accumulation_steps (int, optional): Number of batches to accumulate gradients over
                                                before performing an optimizer step. Defaults to 32.
            epochs (int, optional): Total number of epochs to train. Defaults to 30.
            start_epoch (int, optional): The epoch to start training from (for resuming).
                                         Defaults to 0.
            best_loss (float, optional): The best validation loss achieved so far (for checkpointing).
                                         Defaults to float('inf').
            log_dir (str, optional): Specific directory for TensorBoard logs. If None, a timestamped
                                     directory is created under `log_dir_base`. Defaults to None.
            checkpoint_dir (str, optional): Specific directory for saving model checkpoints. If None,
                                            a timestamped directory is created under `checkpoint_dir_base`.
                                            Defaults to None.
            log_dir_base (str, optional): Base directory for TensorBoard logs.
                                          Defaults to '/media/hoangdv/cv_logs'.
            checkpoint_dir_base (str, optional): Base directory for model checkpoints.
                                                 Defaults to '/media/hoangdv/cv_checkpoints'.
        """
        model.to(self.device) # Ensure model is on the correct device
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # ===  Set up directories for logging and checkpoints  ===
        if log_dir is None:
            log_dir = os.path.join(log_dir_base, f"{self.mode}_{timestamp}") # Add mode to log dir
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(checkpoint_dir_base, f"{self.mode}_{timestamp}") # Add mode to checkpoint dir


        best_checkpoint_path = os.path.join(checkpoint_dir, f'diffusion_model_{self.mode}_best.pth')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # ===  Set up TensorBoard logging  ===
        writer = SummaryWriter(log_dir)

        # ===  Calculate initial global_step if resuming  ===
        effective_batches_per_epoch = len(dataset) // accumulation_steps
        if len(dataset) % accumulation_steps != 0:
            effective_batches_per_epoch +=1


        global_step_optimizer = start_epoch * effective_batches_per_epoch
        batch_step_counter = start_epoch * len(dataset)

        current_accumulation_idx = 0

        print(f"Starting training in '{self.mode}' mode on device: {self.device} with {accumulation_steps} accumulation steps.")
        print(f"Logging to: {log_dir}")
        print(f"Saving checkpoints to: {checkpoint_dir}")
        print(f"Initial global optimizer steps: {global_step_optimizer}, initial batch steps: {batch_step_counter}")

        optimizer.zero_grad()

        for epoch in range(start_epoch, epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            model.train()
            progress_bar = tqdm(total=len(dataset), desc=f"Training ({self.mode})")
            epoch_losses = []

            for (low_res_image_batch, 
                 up_scale_image_batch, 
                 _, 
                 residual_image_0_1_batch) in dataset:
                
                if context == "LR":
                    context = low_res_image_batch.to(self.device)
                elif context == "HR":
                    context = up_scale_image_batch.to(self.device)
                residual_image_0_1_batch = residual_image_0_1_batch.to(self.device)

                actual_batch_size = residual_image_0_1_batch.shape[0]
                t = torch.randint(0, self.timesteps, (actual_batch_size,), device=self.device, dtype=torch.long)
                residual_image_0_1_batch_t, noise_added = self.q_sample(residual_image_0_1_batch, t)

                # Determine the target based on the model's prediction mode
                if self.mode == "v_prediction":
                    # Calculate the target 'v' for v-prediction
                    # v = sqrt(alphas_cumprod_t) * noise - sqrt(1 - alphas_cumprod_t) * x_0
                    sqrt_alphas_cumprod_t_extracted = self._extract(self.sqrt_alphas_cumprod, t, residual_image_0_1_batch_t.shape)
                    sqrt_one_minus_alphas_cumprod_t_extracted = self._extract(self.sqrt_one_minus_alphas_cumprod, t, residual_image_0_1_batch_t.shape)
                    target = sqrt_alphas_cumprod_t_extracted * noise_added - sqrt_one_minus_alphas_cumprod_t_extracted * residual_image_0_1_batch
                elif self.mode == "noise":
                    # Target is the noise itself
                    target = noise_added
                else:
                    # This case should ideally not be reached due to __init__ validation
                    raise ValueError(f"Unsupported training mode: {self.mode}")

                # Forward pass: model predicts based on its mode (either v or noise)
                predicted_output = model(residual_image_0_1_batch_t, t, context=context)

                # Calculate loss
                loss = F.mse_loss(predicted_output, target)
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()

                current_accumulation_idx += 1

                if current_accumulation_idx >= accumulation_steps:
                    optimizer.step()
                    optimizer.zero_grad()
                    current_accumulation_idx = 0
                    global_step_optimizer +=1

                loss_value = loss.detach().item()
                epoch_losses.append(loss_value)

                writer.add_scalar(f'Loss_{self.mode}/batch', loss_value, batch_step_counter)

                batch_step_counter += 1
                progress_bar.update(1)
                progress_bar.set_description(f"Mode: {self.mode} Loss: {loss_value:.4f} OptSteps: {global_step_optimizer}")

            mean_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
            print(f"Epoch {epoch+1} Average Loss ({self.mode}): {mean_loss:.4f}")
            writer.add_scalar(f'Loss_{self.mode}/epoch', mean_loss, epoch + 1)

            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'global_optimizer_steps': global_step_optimizer,
                    'mode': self.mode # Save mode in checkpoint
                }, best_checkpoint_path)
                print(f"Saved new best model checkpoint to {best_checkpoint_path} (Epoch {epoch+1}, Mode: {self.mode}, OptSteps: {global_step_optimizer})")

            # generate sample images for TensorBoard
            if (epoch + 1) % 5 == 0:
                print(f"Generating sample images for TensorBoard at epoch {epoch+1}...")
                generator = ResidualGenerator(
                    img_channels=3,
                    img_size=256,
                    device=self.device,
                    num_train_timesteps=self.timesteps,
                    predict_mode=self.mode
                )
                sample_images = []
                for i in range(3):
                    # load some low resolution images from the dataset
                    low_res_image_batch, up_scale_image_batch, original_image_batch, _ = next(iter(dataset))
                    low_res_image_batch = low_res_image_batch.to(self.device)
                    up_scale_image_batch = up_scale_image_batch.to(self.device)

                    low_res_image = low_res_image_batch[0].unsqueeze(0) # add batch dimension
                    up_scale_image = up_scale_image_batch[0].unsqueeze(0) # add batch dimension

                    # (1, c, h, w)
                    residual = generator.generate_images(model=model, low_resolution_image=low_res_image, num_images=1)
                    image = up_scale_image + residual
                    image = torch.clamp(image, -1.0, 1.0)
                    image = (image + 1.0) / 2.0 # Normalize to [0, 1]
                    # (1, c, h, w) -> (c, h, w)
                    image = image.cpu().numpy().squeeze(0) # Remove batch dimension and move to CPU
                    # save the image to TensorBoard
                    sample_image = (image * 255).astype(np.uint8) # Convert to uint8
                    sample_image = np.clip(sample_image, 0, 255) # Ensure values are in [0, 255]
                    original_image = original_image_batch[0].cpu().numpy()
                    imgs = [low_res_image.squeeze(0).cpu().numpy(), sample_image, original_image]
                    sample_images.append(imgs)

                # Save generated images to TensorBoard
                for i, imgs in enumerate(sample_images):
                    low_res_image, up_scale_image, original_image = imgs
                    low_res_image = (low_res_image + 1.0) / 2.0 # Normalize to [0, 1]
                    low_res_image = low_res_image * 255.0
                    original_image = (original_image + 1.0) / 2.0 # Normalize to [0, 1]
                    original_image = original_image * 255.0
                    low_res_image = low_res_image.astype(np.uint8)
                    original_image = original_image.astype(np.uint8)
                    writer.add_image(f'low_res_{i}', low_res_image, epoch + 1, dataformats='CHW')
                    writer.add_image(f'up_scale_{i}', up_scale_image, epoch + 1, dataformats='CHW')
                    writer.add_image(f'original_{i}', original_image, epoch + 1, dataformats='CHW')

            progress_bar.close()

        if current_accumulation_idx > 0:
            print(f"Performing final optimizer step for {current_accumulation_idx} remaining accumulated gradients...")
            optimizer.step()
            optimizer.zero_grad()
            global_step_optimizer +=1
            print(f"Final gradients applied. Total optimizer steps: {global_step_optimizer}")

        writer.close()
        print(f"Training finished for mode '{self.mode}'.")


    @torch.no_grad()
    def p_sample(self, model, x_t, t_int):
        """
        Performs one step of the reverse diffusion process (denoising) p(x_{t-1} | x_t).
        This version assumes the model predicts the noise `epsilon`.
        It's most suitable if the model was trained in "noise" prediction mode.

        Args:
            model (torch.nn.Module): The neural network model that predicts noise `epsilon`.
            x_t (torch.Tensor): The noisy image at timestep `t`, shape [B, C, H, W].
            t_int (int): The current timestep as an integer.

        Returns:
            torch.Tensor: The denoised image `x_{t-1}`.
        """
        model.eval()
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t_int, dtype=torch.long, device=self.device)

        predicted_noise = model(x_t, t_tensor) # Model is expected to output noise

        beta_t = self.betas[t_int]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_int]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t_int]

        coeff = beta_t / sqrt_one_minus_alpha_cumprod_t
        mean_pred = sqrt_recip_alpha_t * (x_t - coeff * predicted_noise)

        posterior_variance_t = self.posterior_variance[t_int]
        noise_for_sampling = torch.randn_like(x_t) if t_int > 0 else torch.zeros_like(x_t)
        x_t_minus_1 = mean_pred + torch.sqrt(posterior_variance_t) * noise_for_sampling
        x_t_minus_1 = torch.clamp(x_t_minus_1, -1.0, 1.0)
        return x_t_minus_1

    @torch.no_grad()
    def p_sample_v_prediction(self, model, x_t, t_int):
        """
        Performs one step of the reverse diffusion process (denoising) p(x_{t-1} | x_t),
        assuming the model predicts `v` (as in "v-prediction" objective).
        It's most suitable if the model was trained in "v_prediction" mode.

        Args:
            model (torch.nn.Module): The neural network model that predicts `v`.
            x_t (torch.Tensor): The noisy image at timestep `t`, shape [B, C, H, W].
            t_int (int): The current timestep as an integer.

        Returns:
            torch.Tensor: The denoised image `x_{t-1}`.
        """
        model.eval()
        batch_size = x_t.shape[0]
        device = x_t.device
        t_tensor = torch.full((batch_size,), t_int, dtype=torch.long, device=device)

        predicted_v = model(x_t, t_tensor) # Model is expected to output v

        sqrt_alpha_prod_t_extracted = self._extract(self.sqrt_alphas_cumprod, t_tensor, x_t.shape)
        sqrt_one_minus_alpha_prod_t_extracted = self._extract(self.sqrt_one_minus_alphas_cumprod, t_tensor, x_t.shape)

        # Convert predicted_v to predicted_noise
        # predicted_noise = sqrt(alphas_cumprod_t) * predicted_v + sqrt(1 - alphas_cumprod_t) * x_t
        predicted_noise = sqrt_alpha_prod_t_extracted * predicted_v + sqrt_one_minus_alpha_prod_t_extracted * x_t

        alpha_t = self.alphas[t_int].to(device)
        beta_t = self.betas[t_int].to(device)
        sqrt_one_minus_alpha_cumprod_scalar_t = self.sqrt_one_minus_alphas_cumprod[t_int].to(device)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t_int].to(device)

        coeff = beta_t / sqrt_one_minus_alpha_cumprod_scalar_t
        mean_pred = sqrt_recip_alpha_t * (x_t - coeff * predicted_noise)

        posterior_variance_t = self.posterior_variance[t_int].to(device)
        noise_for_sampling = torch.randn_like(x_t) if t_int > 0 else torch.zeros_like(x_t)
        x_t_minus_1 = mean_pred + torch.sqrt(posterior_variance_t) * noise_for_sampling
        x_t_minus_1 = torch.clamp(x_t_minus_1, -1.0, 1.0)
        return x_t_minus_1

    def save_model(self, model, save_path, optimizer=None, epoch=None, loss=None, save_weights_only=False):
        """
        Saves the model and optionally the optimizer state, epoch, loss, and current mode.

        Args:
            model (torch.nn.Module): The PyTorch model to save.
            save_path (str): Path to save the model checkpoint.
            optimizer (torch.optim.Optimizer, optional): The optimizer state to save. Defaults to None.
            epoch (int, optional): The current epoch number. Defaults to None.
            loss (float, optional): The current loss value. Defaults to None.
            save_weights_only (bool, optional): If True, only saves the model's state_dict.
                                                Otherwise, saves a checkpoint dictionary.
                                                Defaults to False.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_weights_only:
            torch.save(model.state_dict(), save_path)
            print(f"Model weights saved to {save_path}")
        else:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'mode': self.mode # Save the current mode
            }
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            if epoch is not None:
                checkpoint['epoch'] = epoch
            if loss is not None:
                checkpoint['loss'] = loss
            # Consider saving global_optimizer_steps if tracked and needed for resume
            # checkpoint['global_optimizer_steps'] = global_optimizer_steps_variable

            torch.save(checkpoint, save_path)
            print(f"Full model checkpoint saved to {save_path}")
            components = [
                "model weights",
                f"mode: {self.mode}",
                "optimizer state" if optimizer else None,
                f"epoch {epoch}" if epoch is not None else None,
                f"loss {loss:.6f}" if loss is not None else None
            ]
            print(f"Saved: {', '.join(c for c in components if c)}")


    def load_model_weights(self, model, model_path, verbose=False):
        """
        Loads model weights from a saved checkpoint file.

        This function can load either a full checkpoint dictionary (extracting
        'model_state_dict') or a raw state_dict file. It handles partial loading
        (missing/unexpected keys) gracefully. It also attempts to load the 'mode'
        attribute if present in the checkpoint and sets it on the instance.

        Args:
            model (torch.nn.Module): The PyTorch model instance to load weights into.
            model_path (str): Path to the model checkpoint file (.pth or .pt).
            verbose (bool, optional): If True, prints detailed information about
                                      missing and unexpected keys. Defaults to False.
        """
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            state_dict_to_load = None
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict_to_load = checkpoint["model_state_dict"]
                else: # Checkpoint is a state_dict itself
                    state_dict_to_load = checkpoint
                
                # Load mode if present in checkpoint and set it on the instance
                if "mode" in checkpoint:
                    loaded_mode = checkpoint["mode"]
                    if loaded_mode in ["v_prediction", "noise"]:
                        self.mode = loaded_mode
                        print(f"DiffusionModel mode set to '{self.mode}' from checkpoint.")
                    else:
                        print(f"Warning: Invalid mode '{loaded_mode}' found in checkpoint. Keeping current mode '{self.mode}'.")
                else:
                    print(f"Info: 'mode' not found in checkpoint. Keeping current mode '{self.mode}'.")
            else: # Loaded object is directly a state_dict
                 state_dict_to_load = checkpoint
                 print(f"Info: Loaded raw state_dict. 'mode' not available in this format. Keeping current mode '{self.mode}'.")


            if state_dict_to_load:
                incompatible_keys = model.load_state_dict(state_dict_to_load, strict=False)
                if incompatible_keys.missing_keys:
                    print(f"Warning: {len(incompatible_keys.missing_keys)} keys in the current model were not found in the checkpoint.")
                    if verbose: print(f"Missing keys: {incompatible_keys.missing_keys}")
                if incompatible_keys.unexpected_keys:
                    print(f"Info: {len(incompatible_keys.unexpected_keys)} keys in the checkpoint were not used by the current model.")
                    if verbose: print(f"Unused (unexpected) keys: {incompatible_keys.unexpected_keys}")

                num_loaded_params = sum(1 for k in state_dict_to_load if k not in incompatible_keys.unexpected_keys and k in model.state_dict())
                print(f"Weights loaded from {model_path}")
                print(f"Successfully loaded {num_loaded_params} compatible parameters into the model.")
            else:
                print(f"Warning: Could not extract state_dict from {model_path}.")

        else:
            print(f"Warning: Model weights path not found: {model_path}. Model weights not loaded.")

    @staticmethod
    def load_checkpoint_for_resume(device, model, optimizer, checkpoint_path):
        """
        Loads a checkpoint for resuming training, including model state, optimizer state,
        epoch number, loss, global optimizer steps, and potentially the training mode.

        If the checkpoint contains a 'mode', it will be printed but not directly set on
        the model instance by this static method. The `DiffusionModel` instance should
        be initialized with the correct mode, or `load_model_weights` (which is an
        instance method) can update the mode if loading weights separately.

        Args:
            device (str or torch.device): The device to load the model and checkpoint onto.
            model (torch.nn.Module): The model instance to load state into.
            optimizer (torch.optim.Optimizer or bnb.optim.Adam8bit): The optimizer instance
                                                                    to load state into.
            checkpoint_path (str): Path to the checkpoint file.

        Returns:
            tuple:
                - int: `start_epoch` (the epoch to resume training from).
                - float: `loaded_loss` (the loss value from the checkpoint).
                - int: `global_optimizer_steps` (the number of optimizer steps from the checkpoint).
                - str or None: `loaded_mode` (the mode string from the checkpoint, or None if not found).
        """
        start_epoch = 0
        loaded_loss = float('inf')
        global_optimizer_steps = 0
        loaded_mode = None # To store the mode from checkpoint

        model.to(device)
        print(f"Ensuring model is on device: {device} for checkpoint loading.")

        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint for resume from: {checkpoint_path} directly onto {device}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                print(f"Checkpoint dictionary loaded successfully to {device} memory.")

                if 'model_state_dict' in checkpoint:
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    if missing_keys: print(f"\nWarning: Missing keys in model state_dict: {missing_keys}")
                    if unexpected_keys: print(f"\nInfo: Unexpected keys in model state_dict from checkpoint: {unexpected_keys}")
                    print(f"Model state loaded successfully onto model on {device}.")
                else:
                    print("Warning: 'model_state_dict' not found in checkpoint. Model weights not loaded.")

                if 'optimizer_state_dict' in checkpoint and optimizer is not None:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        for state in optimizer.state.values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.to(device)
                        print(f"Optimizer state loaded successfully and moved to {device}.")
                    except Exception as optim_load_err:
                         print(f"Error loading optimizer state: {optim_load_err}. Optimizer will start from scratch.")
                elif optimizer is None: print("Warning: Optimizer not provided, skipping optimizer state loading.")
                else: print("Warning: 'optimizer_state_dict' not found in checkpoint. Optimizer starts from scratch.")

                if 'epoch' in checkpoint:
                    saved_epoch = checkpoint['epoch']
                    start_epoch = saved_epoch + 1
                    print(f"Resuming training from epoch: {start_epoch}")
                else: print("Warning: Epoch number not found in checkpoint. Starting from epoch 0.")

                if 'loss' in checkpoint:
                    loaded_loss = checkpoint['loss']
                    print(f"Loaded loss from checkpoint: {loaded_loss:.6f}")
                else: print("Info: Loss value not found in checkpoint. Using default best_loss.")

                if 'global_optimizer_steps' in checkpoint:
                    global_optimizer_steps = checkpoint['global_optimizer_steps']
                    print(f"Loaded global optimizer steps: {global_optimizer_steps}")
                else: print("Info: 'global_optimizer_steps' not found in checkpoint. Starting from 0.")

                if 'mode' in checkpoint: # Load mode
                    loaded_mode = checkpoint['mode']
                    print(f"Loaded mode from checkpoint: '{loaded_mode}'. Ensure DiffusionModel instance is initialized accordingly or use load_model_weights to set it.")
                else:
                    print("Info: 'mode' not found in checkpoint.")


            except Exception as e:
                print(f"Error loading checkpoint: {e}. Training will start from scratch.")
                model.to(device)
                start_epoch = 0; loaded_loss = float('inf'); global_optimizer_steps = 0; loaded_mode = None
        else:
            print(f"Checkpoint file not found at {checkpoint_path}. Training will start from scratch.")
            model.to(device)
            start_epoch = 0; loaded_loss = float('inf'); global_optimizer_steps = 0; loaded_mode = None

        return start_epoch, loaded_loss, global_optimizer_steps, loaded_mode

class ResidualGenerator:
    """
    A class for generating images using a pre-trained diffusion model and a scheduler.

    This class can be configured to work with models that predict either 'v' (v-prediction)
    or 'noise' (epsilon-prediction) by setting the `predict_mode` during
    initialization. It uses a `diffusers.SchedulerMixin` (like DDIMScheduler)
    to perform the reverse diffusion process.

    Attributes:
        img_channels (int): Number of channels in the image (e.g., 3 for RGB).
        img_size (int): Height and width of the image.
        device (str or torch.device): Device for tensor operations.
        num_train_timesteps (int): The number of timesteps the diffusion model was trained for.
                                   This is used to initialize the scheduler correctly.
        predict_mode (str): The prediction type the model is expected to output,
                            and how the scheduler should interpret it ("v_prediction" or "noise").
        betas (torch.Tensor): The beta schedule used for the scheduler.
        scheduler (diffusers.SchedulerMixin): The scheduler instance (e.g., DDIMScheduler)
                                              used for the denoising steps, configured according
                                              to `predict_mode`.
    """
    def __init__(self, 
                 img_channels=3, 
                 img_size=256, 
                 device='cuda', 
                 num_train_timesteps=1000, 
                 predict_mode='v_prediction'):
        """
        Initializes the ResidualGenerator.

        Sets up image parameters, device, and a DDIMScheduler. The scheduler is
        configured based on the `predict_mode` (either "v_prediction"
        or "noise") and uses a cosine beta schedule.

        Args:
            img_channels (int, optional): Number of image channels. Defaults to 3.
            img_size (int, optional): Size (height and width) of the image. Defaults to 32.
            device (str or torch.device, optional): Device for computations. Defaults to 'cuda'.
            num_train_timesteps (int, optional): Number of training timesteps for the
                                                 diffusion model this generator will use.
                                                 Defaults to 1000.
            predict_mode (str, optional): Specifies what the diffusion model is expected to predict.
                                          Can be "v_prediction" or "noise".
                                          Defaults to "v_prediction".
        Raises:
            ValueError: If `predict_mode` is not "v_prediction" or "noise".
        """
        self.img_channels = img_channels
        self.img_size = img_size
        self.device = device
        self.num_train_timesteps = num_train_timesteps

        if predict_mode not in ["v_prediction", "noise"]:
            raise ValueError("Prediction mode must be 'v_prediction' or 'noise'")
        self.predict_mode = predict_mode

        def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
            # Helper function to generate cosine beta schedule
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=dtype)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0., 0.999)

        self.betas = cosine_beta_schedule(self.num_train_timesteps).to(self.device)

        # Initialize the DDIM scheduler based on the predict_mode
        # The `prediction_type` parameter of the scheduler is crucial.
        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            trained_betas=self.betas.cpu().numpy(), # DDIMScheduler might expect numpy array for custom betas
            beta_schedule="trained_betas", # Indicate that we are providing pre-computed betas
            prediction_type=self.predict_mode, # This directly uses the chosen mode
            clip_sample=False, # Typically set to False for v-prediction, final clipping is manual.
                               # For noise prediction, schedulers often handle clipping if set to True,
                               # but False + manual clamp is also a common pattern.
            set_alpha_to_one=False, # Standard for cosine schedules
            steps_offset=1, # A common setting for many schedulers
        )
        # Corrected print statement to reflect the actual configured mode
        print(f"ResidualGenerator initialized with {type(self.scheduler).__name__}, "
              f"configured for model prediction_type='{self.predict_mode}'.")

    @torch.no_grad()
    def generate_images(self, model, low_resolution_image, num_images=1, num_inference_steps=50):
        """
        Generates images using the provided diffusion model and the configured scheduler.

        The model's output (either 'v' or 'noise') should match the
        `predict_mode` this ResidualGenerator was initialized with. The model itself
        (e.g., a U-Net) does not need a '.mode' attribute; this generator handles
        the interpretation of its output based on `self.predict_mode`.

        Args:
            model (torch.nn.Module): The pre-trained diffusion model (e.g., a U-Net).
                                     It should accept `x_t` (noisy image) and `t` (timestep)
                                     as input and output a tensor corresponding to its
                                     training objective (either 'v' or 'noise').
            num_images (int, optional): Number of images to generate. Defaults to 1.
            num_inference_steps (int, optional): Number of denoising steps to perform.
                                                 Fewer steps lead to faster generation but
                                                 potentially lower quality. Defaults to 50.

        Returns:
            torch.Tensor: A batch of generated images, normalized to the [0, 1] range.
                          Shape: [num_images, img_channels, img_size, img_size].
        """
        model.eval() # Set the model to evaluation mode
        model.to(self.device) # Ensure model is on the correct device

        # Updated print statement to include the operating mode
        print(f"Generating {num_images} images using {num_inference_steps} steps "
              f"with {type(self.scheduler).__name__} (model expected to predict: '{self.predict_mode}') "
              f"on device {self.device}...")

        # Set the number of inference steps for the scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # Initialize with random noise (latent space representation)
        # Shape: [batch_size, num_channels, height, width]
        image_latents = torch.randn(
            (num_images, self.img_channels, self.img_size, self.img_size),
            device=self.device
        )

        # Scale the initial noise by the scheduler's init_noise_sigma
        # This is important for some schedulers to ensure the noise is at the right magnitude
        image_latents = image_latents * self.scheduler.init_noise_sigma

        # Iteratively denoise the latents
        for t_step in tqdm(self.scheduler.timesteps, desc="Generating images"):
            # Prepare model input: current noisy latents
            model_input = image_latents

            # The model needs the current latents and the timestep `t_step`
            # Ensure t_step is correctly shaped for the model [batch_size]
            t_for_model = t_step.unsqueeze(0).expand(num_images).to(self.device) # Expand to batch size

            # Model predicts based on its training (either 'v' or 'noise')
            model_output = model(model_input, t_for_model, context=low_resolution_image)

            # Use the scheduler's step function to compute the previous noisy sample
            # The scheduler will interpret `model_output` based on its `prediction_type`
            # (which was set from `self.predict_mode` during initialization).
            scheduler_output = self.scheduler.step(model_output, t_step, image_latents)
            image_latents = scheduler_output.prev_sample

        # Post-processing:
        # The output `image_latents` after the loop is the generated image,
        # typically in the [-1, 1] range if the model was trained on such data.
        # Denormalize to [0, 1] for visualization or saving.
        generated_images = (image_latents + 1.0) / 2.0
        generated_images = torch.clamp(generated_images, 0.0, 1.0) # Clip to ensure valid range

        print("Image generation complete.")
        return generated_images

if __name__ == "__main__":
    # Example usage ResidualGenerator
    generator = ResidualGenerator()
    unet = UNet(base_dim=32)
    low_resolution_image = torch.randn(1, 3, 256, 256) # Example low-res image
    low_resolution_image = low_resolution_image.to('cuda') # Move to GPU if available)
    res = generator.generate_images(unet, low_resolution_image)
    print(res.shape)