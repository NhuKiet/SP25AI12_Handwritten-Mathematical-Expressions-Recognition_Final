from bttr.lit_bttr import LitBTTR
from PIL import Image 
import torch
import cv2
import numpy as np
import os
import glob
from torchvision.transforms import ToTensor
import argparse

def preprocess_image(img_path, mode="minimal"):
    """
    Preprocess image for the BTTR model
    
    Parameters:
    - img_path: Path to the image
    - mode: Preprocessing mode
        - "minimal": Just load and convert to tensor (closest to training)
        - "standard": Basic processing with CLAHE
        - "aggressive": Grayscale + threshold (for handwriting)
    """
    print(f"Processing image: {img_path} with mode: {mode}")
    
    # Load image
    if mode == "minimal":
        # Just load and convert to tensor - minimal processing
        img = Image.open(img_path)
        if img.mode != 'L':
            img = img.convert('L')  # Convert to grayscale as BTTR expects grayscale
        img_tensor = ToTensor()(img)
    
    elif mode == "standard":
        # Standard processing with CLAHE
        img = Image.open(img_path)
        if img.mode != 'L':
            img = img.convert('L')
        
        img_cv = np.array(img)
        
        # Apply CLAHE 
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_cv = clahe.apply(img_cv)
        
        # Convert back to PIL and then to tensor
        img_tensor = ToTensor()(Image.fromarray(img_cv))
    
    else:  # aggressive
        # More aggressive processing for handwriting
        img = Image.open(img_path)
        if img.mode != 'L':
            img = img.convert('L')
        
        img_cv = np.array(img)
        
        # Apply threshold
        _, binary = cv2.threshold(
            img_cv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Convert back to PIL and then to tensor
        img_tensor = ToTensor()(Image.fromarray(binary))
    
    return img_tensor

def find_bttr_checkpoint():
    """Search for BTTR checkpoint in common locations"""
    # Common checkpoint locations
    potential_paths = [
        "BTTR/lightning_logs/version_0/checkpoints/epoch=259-step=97759.ckpt",
        "lightning_logs/version_0/checkpoints/epoch=259-step=97759.ckpt",
        *glob.glob("BTTR/lightning_logs/version_*/checkpoints/*.ckpt"),
        *glob.glob("lightning_logs/version_*/checkpoints/*.ckpt")
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            print(f"Found checkpoint at: {path}")
            return path
    
    return None

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test BTTR model for math expression recognition')
    parser.add_argument('--image', type=str, default="testdoan.jpeg", help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default="aggressive", choices=["minimal", "standard", "aggressive"], 
                        help='Image preprocessing mode')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference even if GPU is available')
    args = parser.parse_args()

    # Use CPU if specified or if no CUDA
    if args.cpu:
        device = torch.device("cpu")
        print("Forcing CPU usage as specified")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        return

    # Find checkpoint if not specified
    ckp_path = args.checkpoint
    if not ckp_path:
        ckp_path = find_bttr_checkpoint()
    
    if not ckp_path or not os.path.exists(ckp_path):
        print("Error: BTTR checkpoint not found.")
        print("Please download the BTTR model checkpoint and either:")
        print("1. Place it at: BTTR/lightning_logs/version_0/checkpoints/epoch=259-step=97759.ckpt")
        print("2. Specify the path with --checkpoint argument")
        return
    
    # Load and preprocess image
    try:
        img = preprocess_image(args.image, mode=args.mode)
    except Exception as e:
        print(f"Error processing image: {e}")
        return
    
    try:
        # Load model and move to device
        print("Loading model...")
        model = LitBTTR.load_from_checkpoint(ckp_path)
        model.eval()
        model.to(device)
        
        # Set CUDA optimizations if device is GPU
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            
        with torch.no_grad():
            # Move input to device
            img = img.to(device)
            
            print("Starting inference on", device)
            # Perform inference using beam search
            hyp = model.beam_search(img)
            
            if hyp:
                print("\nResult:")
                print(f"Predicted LaTeX: {hyp}")
            else:
                print("No result found.")
    
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("GPU out of memory error. Try using CPU with --cpu flag.")
        elif "CUDA error" in str(e):
            print("CUDA error occurred. Check your GPU drivers and CUDA installation.")
        else:
            print(f"Runtime error: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure BTTR is properly installed. Try: pip install -e ./BTTR")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 