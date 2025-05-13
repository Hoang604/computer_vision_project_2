from diffusion import ResidualGenerator, DiffusionModel
from bicubic import upscale_image
import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import argparse
from smaller_unet import UNet
from util import ImageDataset

def downsample_image(img, scale_factor):
    """
    Downsamples an image by a given scale factor using bicubic interpolation.

    Args:
        img (numpy.ndarray): Input image.
        scale_factor (int): The factor by which to downsample the image.

    Returns:
        numpy.ndarray: Downsampled image.
    """
    height, width = img.shape[:2]
    new_size = (width // scale_factor, height // scale_factor)
    downsampled_img = cv.resize(img, new_size, interpolation=cv.INTER_CUBIC)
    return downsampled_img

def upscale_image_with_diffusion(model, image_path, output_path, generator_mode="v_prediction", scale_factor=4):
    """
    Upscales an image using a diffusion model and bicubic interpolation.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the upscaled image.
        scale_factor (int): The factor by which to upscale the image.
    """

    dataset = ImageDataset(folder_path="/home/hoang/python/cv_project_2/data", img_size=256, downscale_factor=4, upscale_function=upscale_image)
    imgs = dataset.__getitem__(1)
    lowres, upscale, img, res = imgs


    context = lowres.to(device="cuda:0").unsqueeze(0)  # Add batch dimension
    # display_lr_encoder_features(model, context, num_features_to_display=32)


    generator = ResidualGenerator(predict_mode=generator_mode)
    residual = generator.generate_residuals(model=model, low_resolution_image=context)
    # print residual statistic
    print("Residual mean: ", residual.mean().item())
    print("Residual std: ", residual.std().item())
    print("Residual min: ", residual.min().item())
    print("Residual max: ", residual.max().item())

    residual = residual.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU
    print("res mean: ", res.mean())
    print("res std: ", res.std())
    print("res min: ", res.min())
    print("res max: ", res.max())
    upscaled_image = upscale + residual


    # Scale to [0, 1] for display
    lowres = lowres.permute(1, 2, 0).cpu().numpy()  # Remove batch dimension and move to CPU
    upscale = upscale.permute(1, 2, 0).cpu().numpy()  # Remove batch dimension and move to CPU
    img = img.permute(1, 2, 0).cpu().numpy()  # Remove batch dimension and move to CPU
    res = res.permute(1, 2, 0).cpu().numpy()  # Remove batch dimension and move to CPU
    upscaled_image = np.transpose(upscaled_image, (1, 2, 0)) 
    residual = np.transpose(residual, (1, 2, 0))
    upscaled_image = (upscaled_image + 1) / 2
    lowres = (lowres + 1) / 2
    upscale = (upscale + 1) / 2
    img = (img + 1) / 2
    res = (res + 1) / 2
    residual = (residual + 1) / 2

    # Save the upscaled image
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 6, 1)
    plt.imshow(lowres)
    plt.title("Low Resolution")
    plt.axis("off")

    plt.subplot(1, 6, 2)
    plt.imshow(upscale)
    plt.title("Bicubic Upscaled")
    plt.axis("off")

    plt.subplot(1, 6, 3)
    plt.imshow(res)
    plt.title("True residual")
    plt.axis("off")
    
    plt.subplot(1, 6, 4)
    plt.imshow(residual)
    plt.title("Predict residual")
    plt.axis("off")

    plt.subplot(1, 6, 5)
    plt.imshow(upscaled_image)
    plt.title("Diffusion upscale")
    plt.axis("off")

    plt.subplot(1, 6, 6)
    plt.imshow(img)
    plt.title("HR image")
    plt.axis("off")

    # Save the figure
    plt.savefig(os.path.join(output_path, "comparison.png"))
    plt.show()
    # Clip the pixel values to be in the valid range


def display_lr_encoder_features(unet_model, low_res_input_tensor, num_features_to_display=8, output_path="."):
    """
    Extracts and displays features from the lr_encoder of the UNet model.

    Args:
        unet_model (torch.nn.Module): The UNet model instance.
        low_res_input_tensor (torch.Tensor): Low-resolution image tensor (B, C, H, W), 
                                                expected to be on the same device as the model.
        num_features_to_display (int): Number of feature maps to display.
        output_path (str): Path to save the feature map visualization.
    """
    unet_model.eval()  # Ensure model is in evaluation mode

    if not hasattr(unet_model, 'lr_encoder'):
        print("Error: The provided UNet model does not have an 'lr_encoder' attribute.")
        return

    try:
        with torch.no_grad():
            features = unet_model.lr_encoder(low_res_input_tensor)
            
    except Exception as e:
        print(f"Error during feature extraction from lr_encoder: {e}")
        return

    # Expected features shape: (B, C_feat, H_feat, W_feat)
    # For visualization, remove batch dim, move to CPU, convert to numpy
    features_np = features.squeeze(0).cpu().numpy()
    print(f"Extracted features shape: {features_np.shape}")

    if features_np.ndim != 3:
        print(f"Warning: Expected 3D features (Channels, Height, Width) after processing, "
                f"but got {features_np.ndim}D shape: {features_np.shape}. "
                f"Cannot display as spatial feature maps.")
        if features_np.ndim == 2 and features_np.shape[0] > 0 and features_np.shape[1] > 0 : # (N, D)
                # Attempt to show as an image if it's 2D (e.g. a matrix of features)
            plt.figure(figsize=(8, 6))
            plt.imshow(features_np, aspect='auto', cmap='viridis')
            plt.title("lr_encoder Features (2D representation)")
            plt.xlabel("Feature Dimension")
            plt.ylabel("Token/Sequence Index")
            plt.colorbar()
            save_filepath = os.path.join(output_path, "lr_encoder_features_2d.png")
        else: # Cannot visualize
            return

    else: # 3D features (C, H, W)
        num_channels = features_np.shape[0]
        print(f"Number of feature maps: {num_channels}")
        num_to_show = min(num_features_to_display, num_channels)

        if num_to_show == 0:
            print("No feature maps to display.")
            return
            
        cols = int(np.ceil(np.sqrt(num_to_show)))
        rows = int(np.ceil(num_to_show / cols))

        plt.figure(figsize=(cols * 2.5, rows * 2.5)) # Adjusted figure size slightly
        plt.suptitle("Features from UNet lr_encoder", fontsize=14)

        for i in range(num_to_show):
            plt.subplot(rows, cols, i + 1)
            feature_map = features_np[i, :, :]
            
            f_min, f_max = feature_map.min(), feature_map.max()
            if abs(f_max - f_min) < 1e-6: # Avoid division by zero for constant feature maps
                feature_map_normalized = np.zeros_like(feature_map) if f_min == 0 else (feature_map - f_min)
            else:
                feature_map_normalized = (feature_map - f_min) / (f_max - f_min)
                
            plt.imshow(feature_map_normalized, cmap='gray')
            plt.title(f"Channel {i+1}", fontsize=10)
            plt.axis("off")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
        save_filepath = os.path.join(output_path, "lr_encoder_features_spatial.png")

    # Save the figure
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    plt.savefig(save_filepath)
    print(f"Saved lr_encoder features visualization to {save_filepath}")
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Upscale an image using diffusion model.")
    parser.add_argument("--image_path", type=str, default="data/test.png", help="Path to the input image.")
    parser.add_argument("--scale_factor", type=int, default=4, help="Scale factor for upscaling.")
    parser.add_argument("--unet_weight", type=str, default="/home/lsdkf;ks;fhoang/python/cv_project_2/cv_checkpoints/v_prediction_20250512-164218/diffusion_model_v_prediction_best.pth", help="Path to the UNet model weights.")
    parser.add_argument("--output_path", type=str, default="diffusion_output/", help="Path to save the upscaled image.")
    parser.add_argument("--mode", type=str, default="v_prediction", choices=["v_prediction", "noise"], help="Mode to use for upscaling.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for computation (cpu or cuda).")

    args = parser.parse_args()

    # Load the model
    model = UNet(base_dim=32,
                 dim_mults=[1, 2, 4, 8],
                 num_resnet_blocks=2,
                 context_dim=256,
                 attn_heads=8)
    
    model.to(args.device)
    DiffusionModel.load_model_weights(args.device, model, model_path=args.unet_weight)
    model.eval()

    upscale_image_with_diffusion(model, args.image_path, args.output_path, args.mode, args.scale_factor)
    
