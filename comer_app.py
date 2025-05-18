from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import sys
import torch
import time
import traceback
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import uuid

# Add CoMER to path
sys.path.append(os.path.abspath('.'))

# Import CoMER model
from CoMER.comer.lit_comer_swin import LitCoMER
from CoMER.comer.datamodule import vocab as comer_vocab

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# CoMER model configuration
COMER_CHECKPOINT = 'CoMER/checkpoints/ComerSwin-epoch=02-val_ExpRate=0.4550.ckpt'
comer_model = None

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_comer_model():
    """Load the CoMER model if not already loaded"""
    global comer_model
    if comer_model is not None:
        return True

    try:
        comer_model = LitCoMER.load_from_checkpoint(COMER_CHECKPOINT)
        comer_model.eval()
        comer_model.to(device)
        print("CoMER model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading CoMER model: {str(e)}")
        traceback.print_exc()
        return False

def preprocess_for_comer(img_path):
    """Preprocess image for the CoMER model"""
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((256, 256), Image.LANCZOS)
    img_tensor = transforms.functional.to_tensor(img_resized)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img_normalized = normalize(img_tensor)
    img_batched = img_normalized.unsqueeze(0)
    return img_batched

def inference_comer(img_tensor):
    """Run inference with the CoMER model"""
    global comer_model
    img_tensor = img_tensor.to(device)
    B, _, H, W = img_tensor.shape
    img_mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)
    
    with torch.no_grad():
        hyps = comer_model.approximate_joint_search(img_tensor, img_mask)
        if hyps:
            best_hyp = hyps[0]
            latex_str = comer_vocab.indices2label(best_hyp.seq)
            return latex_str
        return "No result found"

@app.route('/')
def index():
    return render_template('comer_index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Save the uploaded file with a unique filename
    filename = str(uuid.uuid4()) + secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Load model if not loaded
        if not load_comer_model():
            return jsonify({'error': 'Failed to load CoMER model'}), 500
        
        # Process image
        img_tensor = preprocess_for_comer(filepath)
        latex = inference_comer(img_tensor)
        
        return jsonify({'latex': latex})
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 