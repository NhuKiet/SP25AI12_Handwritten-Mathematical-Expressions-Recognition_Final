from CoMER.comer.lit_comer_swin import LitCoMER
from PIL import Image 
import torch
import cv2
import numpy as np
from torchvision import transforms
from CoMER.comer.datamodule import vocab

def preprocess_image(img_path, mode="minimal"):
    """
    Preprocess image for the CoMER model
    
    Parameters:
    - img_path: Path to the image
    - mode: Preprocessing mode
        - "minimal": Just resize and normalize (closest to training)
        - "standard": Resize + basic CLAHE
        - "aggressive": Grayscale + threshold (for handwriting)
    """
    print(f"Processing image: {img_path} with mode: {mode}")
    
    # Load image
    img = Image.open(img_path).convert("RGB")
    
    # Resize image (always needed)
    img_resized = img.resize((256, 256), Image.LANCZOS)
    
    if mode == "minimal":
        # Just convert to tensor - minimal processing
        img_tensor = transforms.functional.to_tensor(img_resized)
    
    elif mode == "standard":
        # Standard processing with CLAHE
        img_cv = np.array(img_resized)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
        # Apply CLAHE on LAB color space
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        updated_lab = cv2.merge((cl, a, b))
        result = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2BGR)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        img_tensor = transforms.functional.to_tensor(Image.fromarray(result_rgb))
    
    else:  # aggressive
        # More aggressive processing for handwriting
        img_cv = np.array(img_resized)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Convert back to RGB
        result_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        # Convert to tensor
        img_tensor = transforms.functional.to_tensor(Image.fromarray(result_rgb))
    
    # Normalize with ImageNet stats (always needed)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img_normalized = normalize(img_tensor)
    
    # Add batch dimension
    img_batched = img_normalized.unsqueeze(0)
    
    return img_batched

def main():
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")

    # Load and preprocess image with minimal preprocessing
    img_path = "testdoan.jpeg"
    
    # Use minimal preprocessing - closest to what the model was trained with
    img = preprocess_image(img_path, mode="aggressive")
    
    # Load model
    ckp_path = "CoMER/checkpoints/ComerSwin-epoch=02-val_ExpRate=0.4550.ckpt"
    
    try:
        # Load model and move to GPU
        print("Loading model...")
        model = LitCoMER.load_from_checkpoint(ckp_path)
        model.eval()
        model.to(device)
        
        # Set CUDA optimizations if available
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            
        with torch.no_grad():
            # Move input to device
            img = img.to(device)
            
            # Create image mask
            B, _, H, W = img.shape
            img_mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)
            
            print("Starting inference on", device)
            # Perform inference using approximate_joint_search (without extra parameters)
            hyps = model.approximate_joint_search(img, img_mask)
            
            if hyps:
                best_hyp = hyps[0]  # Get the top hypothesis
                latex_str = vocab.indices2label(best_hyp.seq)
                score = best_hyp.score
                
                print(f"Predicted LaTeX: {latex_str}")
                print(f"Score: {score:.4f}")
                
                # Also show some alternative predictions
                if len(hyps) > 1:
                    print("\nAlternative predictions:")
                    for i in range(1, min(3, len(hyps))):
                        alt_latex = vocab.indices2label(hyps[i].seq)
                        alt_score = hyps[i].score
                        print(f"{i}. {alt_latex} (score: {alt_score:.4f})")
            else:
                print("No hypothesis found.")
    
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("GPU out of memory error. Try using a smaller batch size or image size.")
        elif "CUDA error" in str(e):
            print("CUDA error occurred. Check your GPU drivers and CUDA installation.")
        else:
            print(f"Runtime error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()




