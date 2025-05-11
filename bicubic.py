import os
import numpy as np
import cv2
from PIL import Image
from typing import Union, Optional, Any

# Attempt to import torch for type hinting and tensor conversion
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

ALLOWED_SCALE_FACTORS: list[int] = [2, 4, 6, 8, 10]
EPSILON = 1e-5 # For float comparisons

def upscale_image(
    image_source: Union[str, np.ndarray, torch.Tensor],
    scale_factor: int,
    save_image: bool = False,
    output_directory: str = "bicubic_output",
    output_filename_prefix: str = "upscaled"
) -> Optional[Union[np.ndarray, 'torch.Tensor']]:
    """
    Upscales an image using Bicubic interpolation with flexible output typing.

    - Saved images are always uint8 [0,255].
    - Returned NumPy array or Torch Tensor matches input type and data range:
        - Input float [0,1] -> Output float [0,1]
        - Input uint8 [0,255] -> Output uint8 [0,255]
        - Input float [0,255] -> Output float [0,255] (same float type)
        - Input np.ndarray -> Output np.ndarray
        - Input torch.Tensor -> Output torch.Tensor (HWC format, on original device)

    Args:
        image_source (Union[str, np.ndarray, torch.Tensor]):
            The source image.
        scale_factor (int):
            The factor by which to upscale. Must be one of ALLOWED_SCALE_FACTORS.
        save_image (bool, optional):
            If True, save the upscaled image (as uint8 [0,255]). Defaults to False.
        output_directory (str, optional):
            Directory for saved images. Defaults to "bicubic_output".
        output_filename_prefix (str, optional):
            Prefix for saved filenames. Defaults to "upscaled".

    Returns:
        Optional[Union[np.ndarray, torch.Tensor]]:
            The upscaled image, matching input object type (NumPy/Torch) and
            original data type/range characteristics, or None on error.
            Torch Tensors are returned in HWC format.

    Raises:
        ValueError: For invalid scale_factor or unsupported input image properties.
        FileNotFoundError: If image_source path not found.
        ImportError: If torch.Tensor input but torch is not installed.
    """
    if scale_factor not in ALLOWED_SCALE_FACTORS:
        raise ValueError(
            f"scale_factor must be one of {ALLOWED_SCALE_FACTORS}. Got {scale_factor}."
        )

    # --- 1. Input Type and Characteristics Detection ---
    input_is_tensor: bool = False
    original_dtype: Optional[np.dtype] = None
    input_was_float_0_1: bool = False # True if input was float and range was [0,1]
    original_torch_device: Optional[Any] = None # For torch.Tensor
    # original_torch_is_chw: bool = False # For potential CHW tensor output

    img_for_processing: np.ndarray # This will be fed to cv2.resize

    original_filename: Optional[str] = None
    original_stem: str = "input" # Default for array/tensor inputs
    original_ext: str = ".png"

    try:
        if isinstance(image_source, str):
            # Input is a file path
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"Image file not found: {image_source}")
            original_filename = os.path.basename(image_source)
            original_stem = os.path.splitext(original_filename)[0]
            _original_ext_temp = os.path.splitext(original_filename)[1]
            if _original_ext_temp: original_ext = _original_ext_temp

            img_pil = Image.open(image_source)
            # Convert to RGB or Grayscale, common Pillow modes
            if img_pil.mode not in ['RGB', 'L', 'RGBA']:
                if img_pil.mode in ['P', 'CMYK', 'YCbCr']: # Common modes to convert
                    img_pil = img_pil.convert('RGB')
                    print(f"Info: Image file mode was {img_pil.mode}, converted to RGB.")
                else: # Attempt conversion for other modes
                    print(f"Warning: Image file mode {img_pil.mode} is unusual. Attempting RGB conversion.")
                    img_pil = img_pil.convert('RGB')
            
            img_np_loaded = np.array(img_pil)

            if img_np_loaded.ndim == 3 and img_np_loaded.shape[2] == 4: # RGBA
                img_for_processing = cv2.cvtColor(img_np_loaded, cv2.COLOR_RGBA2RGB)
                print("Info: RGBA image loaded, converted to RGB for processing.")
            elif img_np_loaded.ndim == 2 or (img_np_loaded.ndim == 3 and img_np_loaded.shape[2] == 1): # Grayscale
                 if img_np_loaded.ndim == 3 : img_np_loaded = np.squeeze(img_np_loaded, axis=2)
                 img_for_processing = img_np_loaded # Keep as grayscale
            elif img_np_loaded.ndim == 3 and img_np_loaded.shape[2] == 3: # RGB
                img_for_processing = img_np_loaded
            else:
                raise ValueError(f"Loaded image from file has unsupported shape: {img_np_loaded.shape}")

            original_dtype = img_for_processing.dtype # Usually uint8 from Pillow
            # input_was_float_0_1 remains False for typical image files

        elif TORCH_AVAILABLE and isinstance(image_source, torch.Tensor):
            input_is_tensor = True
            tensor_in = image_source.detach()
            original_torch_device = tensor_in.device
            
            # Convert tensor to HWC NumPy array for consistent processing by OpenCV
            np_from_tensor = tensor_in.cpu().numpy()
            
            # Handle NCHW or NHWC (single batch element)
            if np_from_tensor.ndim == 4:
                if np_from_tensor.shape[0] == 1:
                    np_from_tensor = np.squeeze(np_from_tensor, axis=0)
                else:
                    raise ValueError("Batch tensor input not supported. Pass a single image tensor.")

            # Handle CHW to HWC
            if np_from_tensor.ndim == 3 and (np_from_tensor.shape[0] == 1 or np_from_tensor.shape[0] == 3 or np_from_tensor.shape[0] == 4) : # CHW like
                # Potentially store this to revert for tensor output if desired
                # original_torch_is_chw = True
                np_from_tensor = np_from_tensor.transpose(1, 2, 0) # Now HWC
            
            # Now np_from_tensor is HWC
            img_for_processing = np_from_tensor # Work with this HWC numpy array
            original_dtype = img_for_processing.dtype # This is the tensor's fundamental dtype

            if np.issubdtype(original_dtype, np.floating):
                min_val, max_val = img_for_processing.min(), img_for_processing.max()
                if min_val >= (0.0 - EPSILON) and max_val <= (1.0 + EPSILON):
                    input_was_float_0_1 = True
                    # Ensure it's float32 for cv2.resize if not already a supported float
                    if not (original_dtype == np.float32 or original_dtype == np.float64):
                        img_for_processing = img_for_processing.astype(np.float32)
                    img_for_processing = np.clip(img_for_processing, 0.0, 1.0)
                else: # Float but not [0,1], assume [0,255] float or other range
                    input_was_float_0_1 = False
                    if not (original_dtype == np.float32 or original_dtype == np.float64):
                         img_for_processing = img_for_processing.astype(np.float32)
            elif not np.issubdtype(original_dtype, np.uint8): # E.g. uint16, int32
                # For simplicity and cv2 compatibility, convert other int types to float32 for processing
                # The original_dtype is kept, but range might be an issue for exact reconstruction
                print(f"Warning: Input tensor dtype {original_dtype} is not uint8 or common float. "
                      "Converting to float32 for processing. Range assumptions apply for output.")
                max_val = np.iinfo(original_dtype).max if np.issubdtype(original_dtype, np.integer) else img_for_processing.max()
                if max_val > 1.0 + EPSILON : # Likely 0-255 range or higher
                    img_for_processing = (img_for_processing.astype(np.float32) / max_val) * 255.0
                    original_dtype = np.float32 # Effective original type is now float representing 0-255
                    input_was_float_0_1 = False
                else: # Treat as 0-1 range
                    img_for_processing = img_for_processing.astype(np.float32) / max_val if max_val > 0 else img_for_processing.astype(np.float32)
                    original_dtype = np.float32
                    input_was_float_0_1 = True


        elif isinstance(image_source, np.ndarray):
            img_for_processing = image_source.copy() # Work on a copy
            original_dtype = img_for_processing.dtype

            if np.issubdtype(original_dtype, np.floating):
                min_val, max_val = img_for_processing.min(), img_for_processing.max()
                if min_val >= (0.0 - EPSILON) and max_val <= (1.0 + EPSILON):
                    input_was_float_0_1 = True
                    if not (original_dtype == np.float32 or original_dtype == np.float64):
                        img_for_processing = img_for_processing.astype(np.float32)
                    img_for_processing = np.clip(img_for_processing, 0.0, 1.0)
                else:
                    input_was_float_0_1 = False
                    if not (original_dtype == np.float32 or original_dtype == np.float64):
                         img_for_processing = img_for_processing.astype(np.float32)
            elif not np.issubdtype(original_dtype, np.uint8): # E.g. uint16
                print(f"Warning: Input NumPy dtype {original_dtype} is not uint8 or common float. "
                      "Converting to float32 for processing. Range assumptions apply for output.")
                max_val = np.iinfo(original_dtype).max if np.issubdtype(original_dtype, np.integer) else img_for_processing.max()
                if max_val > 1.0 + EPSILON:
                    img_for_processing = (img_for_processing.astype(np.float32) / max_val) * 255.0
                    original_dtype = np.float32 # Effective original type
                    input_was_float_0_1 = False
                else:
                    img_for_processing = img_for_processing.astype(np.float32) / max_val if max_val > 0 else img_for_processing.astype(np.float32)
                    original_dtype = np.float32
                    input_was_float_0_1 = True
            # If uint8, input_was_float_0_1 is False, img_for_processing is fine.

        elif not TORCH_AVAILABLE and type(image_source).__module__ == 'torch':
            raise ImportError("Input is a torch.Tensor, but 'torch' is not installed.")
        else:
            raise TypeError(f"Unsupported image_source type: {type(image_source)}")

        # Ensure img_for_processing has a supported dtype for cv2.resize
        # (uint8, float32, float64 are primary targets from above logic)
        if not (img_for_processing.dtype == np.uint8 or \
                img_for_processing.dtype == np.float32 or \
                img_for_processing.dtype == np.float64):
            print(f"Warning: img_for_processing has dtype {img_for_processing.dtype}. Forcing to float32.")
            img_for_processing = img_for_processing.astype(np.float32)
        
        # Handle image channels (e.g. RGBA to RGB, consistent grayscale)
        if img_for_processing.ndim == 3 and img_for_processing.shape[2] == 4: # RGBA
            img_for_processing = cv2.cvtColor(img_for_processing, cv2.COLOR_RGBA2RGB)
        elif img_for_processing.ndim == 3 and img_for_processing.shape[2] == 1: # Grayscale (H,W,1)
            img_for_processing = np.squeeze(img_for_processing, axis=2) # to (H,W)
        
        if img_for_processing is None or img_for_processing.size == 0:
            raise ValueError("Image data is empty or invalid after initial processing.")

    except Exception as e:
        print(f"Error during input processing: {e}")
        return None

    # --- 2. Perform Upscaling ---
    try:
        if img_for_processing.ndim == 2: # Grayscale (H, W)
            h, w = img_for_processing.shape
        elif img_for_processing.ndim == 3: # Color (H, W, C)
            h, w, c = img_for_processing.shape
        else:
            raise ValueError(f"Image for processing has unexpected dimensions: {img_for_processing.ndim}")

        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        # cv2.resize preserves dtype of img_for_processing
        upscaled_intermediate_np = cv2.resize(img_for_processing, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        # If input was (H,W), output is (new_H, new_W). If (H,W,C), output is (new_H, new_W, C)
    except cv2.error as e:
        print(f"OpenCV error during resizing: {e}")
        return None
    except Exception as e:
        print(f"Error during upscaling step: {e}")
        return None

    # --- 3. Prepare Return Value (matching input type/range) ---
    result_image: Union[np.ndarray, 'torch.Tensor']
    # upscaled_intermediate_np has same dtype as img_for_processing
    
    # Make a copy to modify for the return value
    processed_result_np = upscaled_intermediate_np.copy()

    if input_was_float_0_1:
        # Input was float [0,1]. Output should be float [0,1] with original float type.
        processed_result_np = np.clip(processed_result_np, 0.0, 1.0)
        if np.issubdtype(original_dtype, np.floating):
            processed_result_np = processed_result_np.astype(original_dtype)
        else: # Fallback if original_dtype wasn't float (should be rare here)
            processed_result_np = processed_result_np.astype(np.float32)
    else: # Input was in 0-255 range (uint8 or float) or other converted type
        processed_result_np = np.clip(processed_result_np, 0, 255) # Clip first
        if np.issubdtype(original_dtype, np.uint8):
            processed_result_np = processed_result_np.astype(np.uint8)
        elif np.issubdtype(original_dtype, np.floating): # Original was float (e.g. [0,255] range)
            processed_result_np = processed_result_np.astype(original_dtype)
        else: # Original was other int type, or file input (defaulting to uint8 for 0-255 range)
            # This path implies original_dtype might have been lost if it wasn't uint8/float.
            # Default to uint8 for integer-like 0-255 outputs.
            processed_result_np = processed_result_np.astype(np.uint8)
            if original_dtype is not None and not np.issubdtype(original_dtype, np.uint8):
                 print(f"Info: Original input dtype {original_dtype} (0-255 range) is returned as uint8.")


    if input_is_tensor:
        # Convert HWC NumPy array back to Torch tensor
        # If original tensor was CHW, user might need to permute. This returns HWC tensor.
        # Example for CHW output: if original_torch_is_chw: processed_result_np = processed_result_np.transpose(2,0,1)
        result_image = torch.from_numpy(processed_result_np.copy()).to(original_torch_device) # .copy() for safety
    else:
        result_image = processed_result_np

    # --- 4. Save Output Image (always as uint8 [0,255]) ---
    if save_image:
        try:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory, exist_ok=True)

            output_filename = f"{output_filename_prefix}_{original_stem}_x{scale_factor}{original_ext}"
            save_path = os.path.join(output_directory, output_filename)

            # Prepare image for saving (uint8, 0-255) from upscaled_intermediate_np
            img_to_save_np: np.ndarray
            temp_save_img = upscaled_intermediate_np.copy()

            if np.issubdtype(temp_save_img.dtype, np.floating):
                # Check if the intermediate float image was effectively [0,1]
                # This can be inferred from how img_for_processing was prepared
                min_proc_val = img_for_processing.min()
                max_proc_val = img_for_processing.max()
                is_intermediate_float_0_1 = (min_proc_val >= (0.0-EPSILON) and max_proc_val <= (1.0+EPSILON) and \
                                            np.issubdtype(img_for_processing.dtype, np.floating) )

                if is_intermediate_float_0_1 :
                    img_to_save_np = (np.clip(temp_save_img, 0.0, 1.0) * 255).astype(np.uint8)
                else: # Assumed float [0,255] or other float range
                    img_to_save_np = np.clip(temp_save_img, 0, 255).astype(np.uint8)
            elif np.issubdtype(temp_save_img.dtype, np.uint8):
                img_to_save_np = np.clip(temp_save_img,0,255) # Already uint8, ensure clip
            else: # Other integer types (e.g., from direct cv2.resize if input was int16)
                print(f"Warning: Intermediate image for saving has dtype {temp_save_img.dtype}. Normalizing and converting to uint8 [0,255].")
                # General normalization for unknown integer types
                min_val, max_val = temp_save_img.min(), temp_save_img.max()
                if max_val == min_val: temp_save_img_float = np.zeros_like(temp_save_img, dtype=np.float32)
                else: temp_save_img_float = (temp_save_img.astype(np.float32) - min_val) / (max_val - min_val)
                img_to_save_np = (np.clip(temp_save_img_float,0.0,1.0) * 255).astype(np.uint8)


            # Determine Pillow mode
            if img_to_save_np.ndim == 2:
                pil_mode = 'L' # Grayscale
            elif img_to_save_np.ndim == 3 and img_to_save_np.shape[2] == 3:
                pil_mode = 'RGB' # Color
            else:
                raise ValueError(f"Cannot save image with shape {img_to_save_np.shape}. Expected 2D or 3D (H,W,3).")

            img_to_save_pil = Image.fromarray(img_to_save_np, mode=pil_mode)
            img_to_save_pil.save(save_path)
            print(f"Upscaled image saved to: {save_path}")

        except Exception as e:
            print(f"Error saving image: {e}")
            # Continue to return result_image even if saving fails

    return result_image

# --- Example Usage (similar to previous, but using upscale_image_v2) ---
if __name__ == "__main__":
    img_path = "data/test.png"
    scale_factor = 2
    save_output = True
    output_dir = "output"
    output_prefix = "upscaled"
    result = upscale_image(
        img_path,
        scale_factor,
        save_image=save_output,
        output_directory=output_dir,
        output_filename_prefix=output_prefix
    )
    if result is not None:
        print("Upscaling completed successfully.")
    else:
        print("Upscaling failed.")
        