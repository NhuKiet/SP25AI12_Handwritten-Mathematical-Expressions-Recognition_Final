import cv2
import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from skimage.morphology import skeletonize

# Add CoMER to path
sys.path.append(os.path.abspath('.'))

# Import CoMER model
from CoMER.comer.lit_comer_swin import LitCoMER
from CoMER.comer.datamodule import vocab as comer_vocab

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CoMER model configuration
COMER_CHECKPOINT = 'CoMER/checkpoints/ComerSwin-epoch=02-val_ExpRate=0.4550.ckpt'

def load_model():
    """Load the CoMER model"""
    print("Loading CoMER model...")
    model = LitCoMER.load_from_checkpoint(COMER_CHECKPOINT)
    model.eval()
    model.to(device)
    print("Model loaded successfully")
    return model

def enhance_image(img_path, show_steps=True):
    """
    Apply advanced image processing to enhance math expressions.
    Returns the processed image as a PIL Image.
    """
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Store processing steps for visualization
    steps = [("Original", gray)]
    
    # Deskew
    try:
        coords = np.column_stack(np.where(gray < 255))
        if len(coords) > 0:  # Only deskew if there are non-white pixels
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = gray.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            steps.append(("Deskewed", gray))
    except Exception as e:
        print(f"Skipping deskew due to error: {e}")
    
    # Adaptive threshold
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 2
    )
    steps.append(("Thresholded", th))
    
    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    steps.append(("Cleaned", clean))
    
    # Skeletonize
    skeleton = skeletonize(clean.astype(bool))
    skeleton = (skeleton * 255).astype(np.uint8)
    steps.append(("Skeletonized", skeleton))
    
    # Dilation for uniform stroke
    stroke_width = 2
    sw_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke_width, stroke_width))
    printed = cv2.dilate(skeleton, sw_kernel, iterations=1)
    steps.append(("Dilated", printed))
    
    # Smooth edges
    blur = cv2.GaussianBlur(printed, (3, 3), 0)
    _, final = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    steps.append(("Final", final))
    
    # Convert to RGB for compatibility with CoMER
    final_rgb = cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)
    
    # Display processing steps
    if show_steps:
        plt.figure(figsize=(15, 10))
        for i, (title, image) in enumerate(steps):
            plt.subplot(3, 3, i+1)
            plt.title(title)
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Convert to PIL image for CoMER preprocessing
    return Image.fromarray(final_rgb)

def preprocess_for_comer(pil_img):
    """Preprocess the image for CoMER model"""
    img_resized = pil_img.resize((256, 256), Image.LANCZOS)
    img_tensor = transforms.functional.to_tensor(img_resized)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img_normalized = normalize(img_tensor)
    img_batched = img_normalized.unsqueeze(0)
    return img_batched

def recognize_math(img_path, use_enhancement=True, show_steps=True):
    """
    Recognize mathematical expression in an image.
    Parameters:
    - img_path: Path to the image file
    - use_enhancement: Whether to use enhanced preprocessing (fixbugs.py)
    - show_steps: Whether to display processing steps
    """
    # Load and preprocess image
    if use_enhancement:
        # Use enhanced preprocessing
        enhanced_img = enhance_image(img_path, show_steps)
        img_tensor = preprocess_for_comer(enhanced_img)
    else:
        # Use standard preprocessing
        img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess_for_comer(img)
    
    # Load model
    model = load_model()
    
    # Run inference
    print("Running inference...")
    img_tensor = img_tensor.to(device)
    B, _, H, W = img_tensor.shape
    img_mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)
    
    with torch.no_grad():
        hyps = model.approximate_joint_search(img_tensor, img_mask)
        if hyps:
            best_hyp = hyps[0]
            latex_str = comer_vocab.indices2label(best_hyp.seq)
            print(f"Predicted LaTeX: {latex_str}")
            
            # Also show some alternative predictions
            if len(hyps) > 1:
                print("\nAlternative predictions:")
                for i in range(1, min(3, len(hyps))):
                    alt_latex = comer_vocab.indices2label(hyps[i].seq)
                    alt_score = hyps[i].score
                    print(f"{i}. {alt_latex} (score: {alt_score:.4f})")
            
            return latex_str
        else:
            print("No result found.")
            return None

if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Math Expression Recognition with Enhanced Preprocessing')
    parser.add_argument('--image', type=str, default="testdoan.jpeg", help='Path to input image')
    parser.add_argument('--no-enhance', action='store_true', help='Disable enhanced preprocessing')
    parser.add_argument('--no-steps', action='store_true', help='Hide preprocessing steps')
    args = parser.parse_args()
    
    # Run recognition
    recognize_math(args.image, not args.no_enhance, not args.no_steps) 