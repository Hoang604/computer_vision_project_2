# Computer Vision Super-Resolution Project (DDPM)

This project implements a Denoising Diffusion Probabilistic Model (DDPM) for image super-resolution. The goal is to upscale low-resolution images to higher resolutions while generating realistic details. The model utilizes a U-Net architecture with attention mechanisms and is trained to predict either the noise added during the forward diffusion process or a "v-prediction" target.

## Core Components

The project is structured into several key Python files:

*   **`diffusion.py`**:
    *   `DiffusionModel`: Implements the core DDPM logic, including the forward (noising) and reverse (denoising) processes. It handles the cosine noise schedule, training loop (with gradient accumulation, TensorBoard logging, checkpointing), and model saving/loading. Supports "v-prediction" and "noise" prediction modes.
    *   `ResidualGenerator`: A class for generating high-resolution images from low-resolution inputs using a trained diffusion model and a scheduler (e.g., `DDIMScheduler` from the `diffusers` library). It can operate in "v-prediction" or "noise" mode based on the trained model.

*   **`unet.py`**:
    *   `UNet`: Defines the U-Net architecture, which serves as the noise predictor in the diffusion model. It incorporates residual blocks, sinusoidal position embeddings for time steps, and attention mechanisms (`BasicTransformerBlock`) for improved feature extraction and context integration.
    *   `BasicTransformerBlock`: A transformer block that combines self-attention and conditional cross-attention (if context is provided) with feed-forward layers.
    *   `ImageContextExtractor`: Extracts contextual features from an image, which can be used for conditioning the U-Net (e.g., using the low-resolution image as context).

*   **`train_diffusion.py`**:
    *   The main script for training the diffusion model. It handles dataset loading, model initialization, optimizer setup (using `bitsandbytes` for 8-bit AdamW), and orchestrates the training process using the `DiffusionModel` class. It includes argument parsing for configuring training parameters.

*   **`util.py`**:
    *   `ImageDataset`: A PyTorch `Dataset` class for loading and preprocessing images. It generates low-resolution images, corresponding upscaled versions (using a provided upscale function like bicubic), the original high-resolution target, and the residual between the original and the upscaled image. It also supports data augmentation (horizontal flipping).
    *   `SinusoidalPositionEmbeddings`: Generates sinusoidal embeddings for time steps, used to inform the U-Net about the current noise level.
    *   `ResidualBlock`: A standard residual block with GroupNorm, SiLU activation, and optional time embedding conditioning, used within the U-Net.
    *   `MultiHeadAttentionBlock`: A self-attention block (though `BasicTransformerBlock` in `unet.py` seems to be the primary one used for more advanced attention).

*   **`bicubic.py`**:
    *   `upscale_image`: A utility function to upscale images using bicubic interpolation. It can handle various input types (file paths, NumPy arrays, PyTorch tensors) and supports different scale factors. This is likely used by `ImageDataset` to create the upscaled version of the low-resolution image for training or as a baseline.

## Features

*   **Denoising Diffusion Probabilistic Model (DDPM)** for image super-resolution.
*   Supports two prediction modes for the diffusion model:
    *   **Noise prediction**: The model learns to predict the noise added to the image.
    *   **V-prediction**: The model learns to predict a target 'v' related to both the noise and the clean image.
*   **U-Net Architecture**: A robust U-Net with:
    *   Residual Blocks for stable training.
    *   Sinusoidal Time Embeddings to condition on noise levels.
    *   Self-Attention and Cross-Attention (`BasicTransformerBlock`) for capturing global dependencies and integrating contextual information (e.g., from the low-resolution image).
*   **Cosine Noise Schedule**: For defining the variance of noise added at each diffusion timestep.
*   **Conditional Generation**: The U-Net can be conditioned on context (e.g., low-resolution image features extracted by `ImageContextExtractor`).
*   **Training Script (`train_diffusion.py`)**:
    *   Configurable training parameters via command-line arguments.
    *   Gradient accumulation to simulate larger batch sizes.
    *   Integration with `bitsandbytes` for memory-efficient 8-bit optimizers (AdamW8bit).
    *   TensorBoard logging for monitoring training progress (loss, generated image samples).
    *   Checkpointing to save the best model and resume training.
*   **Flexible Image Dataset (`ImageDataset`)**:
    *   Loads images and prepares low-resolution inputs, bicubic upscaled versions, and high-resolution targets.
    *   Calculates the residual image (difference between HR original and bicubic upscaled LR).
    *   Supports data augmentation (horizontal flipping).
    *   Normalizes images to the `[-1, 1]` range.
*   **Bicubic Upscaling Utility (`bicubic.py`)**: A standalone module for performing bicubic interpolation, useful for data preparation or as a baseline comparison.
*   **Inference/Sampling (`ResidualGenerator`)**:
    *   Uses `diffusers.DDIMScheduler` for efficient sampling.
    *   Can generate images based on a trained model operating in either "v-prediction" or "noise" mode.

## Setup and Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <https://github.com/Hoang604/computer_vision_project_2.git>
    cd <https://github.com/Hoang604/computer_vision_project_2.git>
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    or if you use conda
    ```bash
    conda create -n env
    conda activate env
    ```

3.  **Install dependencies:**
    You will need Python 3.x and the following core libraries. You can install them using pip:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Or your specific CUDA version
    pip install numpy opencv-python Pillow tqdm bitsandbytes diffusers tensorboard torchinfo
    ```
    or you can directly install all dependencies using `pip install -r requirements.txt`

## Usage

### 1. Prepare your Dataset

*   Place your high-resolution training images in a single folder.
*   The `ImageDataset` in [util.py](http://_vscodecontentref_/1) will automatically create low-resolution and other variants during training.
*   Update the default `image_folder` path in [train_diffusion.py](http://_vscodecontentref_/2) or provide it as a command-line argument.

### 2. Training the Diffusion Model

The [train_diffusion.py](http://_vscodecontentref_/3) script is used to train the model. Here are some of the key command-line arguments:

*   `--image_folder`: Path to your training image dataset.
*   `--img_size`: Target size for the high-resolution images (e.g., 256 for 256x256).
*   `--downscale_factor`: Factor by which to downscale the original image to create the low-resolution input.
*   `--epochs`: Number of training epochs.
*   `--batch_size`: Batch size per device.
*   `--accumulation_steps`: Gradient accumulation steps. Effective batch size will be `batch_size * accumulation_steps`.
*   `--learning_rate`: Learning rate for the optimizer.
*   `--timesteps`: Number of diffusion timesteps.
*   `--diffusion_mode`: Prediction mode for the diffusion model. Options: `"v_prediction"` or `"noise"`.
*   `--unet_base_dim`: Base channel dimension for the U-Net.
*   `--unet_dim_mults`: Channel multipliers for each U-Net resolution level (e.g., `1 2 4`).
*   `--base_log_dir`: Base directory for logging
*   `--base_checkpoint_dir`: Base directory for saving checkpoint
*   `--continue_log_dir`: Directory for continue logging on old TensorBoard logs.
*   `--continue_checkpoint_dir`: Directory for continue training on old checkpoints.
*   `--weights_path`: Path to a pre-trained model checkpoint to resume training.

**Example training command:**

```bash
python train_diffusion.py \
    --image_folder /path/to/your/images \
    --img_size 256 \
    --downscale_factor 4 \
    --batch_size 8 \
    --accumulation_steps 8 \
    --learning_rate 1e-5 \
    --epochs 100 \
    --diffusion_mode v_prediction \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints