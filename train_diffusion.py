import torch
from torch.utils.data import DataLoader
from util import ImageDataset
import os
import bitsandbytes as bnb
import argparse
from unet import UNet
from diffusion import DiffusionModel

            
def train_diffusion(args):
    """
    Main function to set up and run the diffusion model training.
    Uses a hardcoded path for the dataset image folder and the simplified Dataset class.
    """

    context = args.context
    assert context in ['LR', 'HR'], "Context must be either 'LR' or 'HR'."
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Create the dataset with the specified folder
    folder_path = args.image_folder if args.image_folder else '/media/tuannl1/heavy_weight/data/cv_data/images256x256'
    train_dataset = ImageDataset(folder_path=folder_path, img_size=args.img_size, downscale_factor=args.downscale_factor)
    print(f"Loaded {len(train_dataset)} images from /media/tuannl1/heavy_weight/data/cv_data/images256x256")

    # Create the DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    diffusion_helper = DiffusionModel(
        timesteps=args.timesteps,
        device=device,
        mode=args.diffusion_mode if args.diffusion_mode else 'v_prediction',
    )
    unet_model = UNet(
        in_channels=args.img_channels,
        out_channels=args.img_channels,
        base_dim=args.unet_base_dim,
        dim_mults=tuple(args.unet_dim_mults),
        num_resnet_blocks=1,
        context_dim=256,
        attn_heads=8
    ).to(device)
    print(f"Front UNet model initialized with base_dim={args.unet_base_dim}, dim_mults={tuple(args.unet_dim_mults)}")
    print("Initializing model with random weights")

    optimizer = bnb.optim.AdamW8bit(
        unet_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    print(f"Optimizer: AdamW, LR: {args.learning_rate}, Weight Decay: {args.weight_decay}") 

    if args.weights_path is not None and os.path.exists(args.weights_path):
        start_epoch, best_loss = diffusion_helper.load_checkpoint_for_resume(device='cuda', model=unet_model, optimizer=optimizer, checkpoint_path=args.weights_path)
        print(f"Loaded model weights from {args.weights_path}") 
        
    else:
        start_epoch = 0
        best_loss = float('inf')
        print("No pre-trained weights found. Starting from scratch.") 

    # --- Start Training ---
    print("\nStarting training process...") 
    try:
        diffusion_helper.train(
            dataset=train_loader,
            model=unet_model,
            optimizer=optimizer,
            accumulation_steps=args.accumulation_steps,
            epochs=args.epochs,
            start_epoch=start_epoch,
            best_loss=best_loss,
            context=context,
            log_dir=args.continue_log_dir if args.continue_log_dir else None,
            checkpoint_dir=args.continue_checkpoint_dir if args.continue_checkpoint_dir else None,
            log_dir_base=args.base_log_dir,
            checkpoint_dir_base=args.base_checkpoint_dir
        )
    except Exception as train_error:
        # Catch potential errors during training
        print(f"\nERROR occurred during training: {train_error}")
        print("This might be due to issues reading image files or shape mismatches.") 

# --- Script Entry Point ---
if __name__ == "__main__": 
    # Set up argument parser (same as before, without --image_folder)
    parser = argparse.ArgumentParser(description="Train Diffusion Model (Simplified Dataset, Hardcoded Path)")
    # Dataset args
    parser.add_argument('--img_size', type=int, default=256, help='Target image size')
    parser.add_argument('--img_channels', type=int, default=3, help='Number of image channels')
    parser.add_argument('--image_folder', type=str, default="/media/tuannl1/heavy_weight/data/cv_data/images256x256", help='Path to the image folder (hardcoded in the script)')
    parser.add_argument('--downscale_factor', type=int, default=4, help='Downscale factor for original image size')
    # Training args
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size') 
    parser.add_argument('--accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Optimizer weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker processes')
    # Diffusion args
    parser.add_argument('--context', type=str, default='LR', help='Context type (LR or HR)')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--weights_path', type=str, default=None, help='Path to pre-trained model weights')
    parser.add_argument('--diffusion_mode', type=str, default='v_prediction', help='Diffusion mode (v_prediction or noise)')
    # UNet args
    parser.add_argument('--unet_base_dim', type=int, default=32, help='Base channel dimension for UNet')
    parser.add_argument('--unet_dim_mults', type=int, nargs='+', default=[1, 2, 4, 8], help='Channel multipliers for UNet')
    # Logging/Saving args
    parser.add_argument('--base_log_dir', type=str, default='/media/hoangdv/cv_logs', help='Base directory for logging')
    parser.add_argument('--base_checkpoint_dir', type=str, default='/media/hoangdv/cv_checkpoints', help='Base directory for saving checkpoints')
    parser.add_argument('--continue_log_dir', type=str, default='/media/hoangdv/cv_logs', help='Directory for continue logging on old TensorBoard logs')
    parser.add_argument('--continue_checkpoint_dir', type=str, default='/media/hoangdv/cv_checkpoints/diffusion_model_v_prediction_best.pth', help='Directory for continue training on old checkpoints')
    # Loading model args
    parser.add_argument('--verbose', action='store_true', help='Print detailed information about weight loading')

    args = parser.parse_args()

    # --- Print Configuration ---
    effective_batch_size = args.batch_size * args.accumulation_steps
    print(f"--- Configuration ---") 
    # Image folder path is hardcoded and printed inside main()
    print(f"Image Size: {args.img_size}x{args.img_size}") 
    print(f"Image Channels: {args.img_channels}") 
    print(f"Diffusion Mode: {args.diffusion_mode}") 
    print(f"Context type: {args.context}")
    print(f"Batch Size (per device): {args.batch_size}") 
    print(f"Accumulation Steps: {args.accumulation_steps}") 
    print(f"Effective Batch Size: {effective_batch_size}") 
    print(f"Epochs: {args.epochs}") 
    print(f"Learning Rate: {args.learning_rate}") 
    print(f"Timesteps: {args.timesteps}") 
    print(f"UNet Base Dim: {args.unet_base_dim}") 
    print(f"UNet Dim Mults: {tuple(args.unet_dim_mults)}") 
    print(f"Log Directory Base: {args.log_dir}") 
    print(f"Checkpoint Directory Base: {args.checkpoint_dir}") 
    print(f"--------------------") 

    # Call the main function
    train_diffusion(args)