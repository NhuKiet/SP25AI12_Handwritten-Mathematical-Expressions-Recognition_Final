from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import sys
import torch
import traceback
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import uuid

# Add all necessary module directories to path
current_dir = os.path.abspath('.')
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'SwinCoMER'))

# Suppress PyTorch Lightning warnings (there are version mismatches)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Multiple.*ModelCheckpoint.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Lightning automatically upgraded.*")

# Track available models
AVAILABLE_MODELS = {}

# Import models
try:
    from SwinCoMER.comer.lit_comer_swin import LitCoMER as SwinCoMER_LitCoMER_Class
    from SwinCoMER.comer.datamodule import vocab as SwinCoMER_vocab_module
    AVAILABLE_MODELS['swincomer'] = True
    print("SwinCoMER model is available")
except ImportError as e:
    AVAILABLE_MODELS['swincomer'] = False
    print(f"Warning: SwinCoMER module not found or has errors: {str(e)}")

if not any(AVAILABLE_MODELS.values()):
    print("WARNING: No models are available. The app will run but won't be able to process images.")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    'swincomer': {
        'checkpoint': 'SwinCoMER/checkpoints/ComerSwin-epoch=02-val_ExpRate=0.4550.ckpt',
        'loaded': False,
        'model': None
    }
}

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Memory management function
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
              f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model(model_name, use_cpu=False):
    """Load a specific model if not already loaded"""
    # Check if model is available
    if not AVAILABLE_MODELS.get(model_name, False):
        print(f"Model {model_name} is not available")
        return False

    if MODEL_CONFIGS[model_name]['loaded']:
        return True

    try:
        # Clear GPU memory before loading model
        clear_gpu_memory()

        # Determine device to use
        target_device = torch.device('cpu') if use_cpu else device

        if model_name == 'swincomer':
            MODEL_CONFIGS[model_name]['model'] = SwinCoMER_LitCoMER_Class.load_from_checkpoint(
                MODEL_CONFIGS[model_name]['checkpoint'],
                map_location=target_device
            )

        MODEL_CONFIGS[model_name]['model'].eval()

        # Move model to device if not already there
        if not use_cpu and target_device.type == 'cuda':
            try:
                MODEL_CONFIGS[model_name]['model'].to(target_device)
            except torch.cuda.OutOfMemoryError:
                print(f"CUDA out of memory when moving {model_name} to GPU. Using CPU instead.")
                MODEL_CONFIGS[model_name]['model'].to("cpu")

        MODEL_CONFIGS[model_name]['loaded'] = True
        print(f"Model {model_name} loaded successfully on {next(MODEL_CONFIGS[model_name]['model'].parameters()).device}")
        return True
    except Exception as e:
        print(f"Error loading {model_name} model: {str(e)}")
        traceback.print_exc()
        return False

# Image preprocessing functions for each model
#
# The training pipeline (SwinCoMER/comer/datamodule/transforms.py) is:
#   PIL.convert("L") -> np.array -> A.Resize(256, 256) -> ToTensorV2()
# Albumentations' ToTensorV2 does NOT divide by 255 and there is no A.Normalize
# anywhere in that pipeline, so the model was trained on float values in
# [0, 255] with three identical (grayscale) channels.
#
# The original demo instead used to_tensor() -> [0, 1] -> ImageNet Normalize on
# a real RGB image, which the model never saw during training. Swin V2 applies a
# LayerNorm right after patch embedding, which absorbs most of a uniform affine
# rescale, so the mismatch is not catastrophic - but it is still train/serve
# skew.
#
# Set this to False to restore the original behaviour and A/B the two paths on
# real samples.
MATCH_TRAINING_PREPROCESSING = True


def preprocess_for_swincomer(img_path):
    if not MATCH_TRAINING_PREPROCESSING:
        # Legacy path: ImageNet-normalized RGB.
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((256, 256), Image.LANCZOS)
        img_tensor = transforms.functional.to_tensor(img_resized)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        return normalize(img_tensor).unsqueeze(0)

    # Mirror the training transform exactly, including cv2's INTER_LINEAR
    # (A.Resize's default) rather than PIL's LANCZOS.
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    arr = cv2.resize(arr, (256, 256), interpolation=cv2.INTER_LINEAR)

    img_tensor = torch.from_numpy(arr).float()   # [H, W], values in [0, 255]
    img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
    return img_tensor.unsqueeze(0)               # [1, 3, 256, 256]

# Model inference functions
def inference_swincomer(img_tensor):
    model = MODEL_CONFIGS['swincomer']['model']
    model_device = next(model.parameters()).device
    img_tensor = img_tensor.to(model_device)
    B, _, H, W = img_tensor.shape
    img_mask = torch.zeros((B, H, W), dtype=torch.bool, device=model_device)

    with torch.no_grad():
        hyps = model.approximate_joint_search(img_tensor, img_mask)
        if hyps:
            best_hyp = hyps[0]
            latex_str = SwinCoMER_vocab_module.indices2label(best_hyp.seq)
            return latex_str
        return "No result found"

@app.route('/')
def index():
    # Pass available models to the template
    return render_template('index.html', available_models=AVAILABLE_MODELS)

@app.route('/available_models')
def get_available_models():
    """API endpoint to get available models"""
    return jsonify(AVAILABLE_MODELS)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    # Get selected models
    selected_models = request.form.getlist('models')
    if not selected_models:
        return jsonify({'error': 'No models selected'}), 400

    # Check if CPU mode is requested
    use_cpu = request.form.get('use_cpu', 'false').lower() == 'true'
    if use_cpu:
        print("CPU mode enabled by user")

    # Filter out unavailable models
    selected_models = [model for model in selected_models if AVAILABLE_MODELS.get(model, False)]

    if not selected_models:
        return jsonify({'error': 'None of the selected models are available'}), 400

    # Save the uploaded file with a unique filename
    filename = str(uuid.uuid4()) + secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    results = {}

    try:
        # Process with each selected model
        for model_name in selected_models:
            if model_name not in MODEL_CONFIGS:
                results[model_name] = {'error': f'Unknown model: {model_name}'}
                continue

            # Load model if not loaded
            if not load_model(model_name, use_cpu):
                results[model_name] = {'error': f'Failed to load model: {model_name}'}
                continue

            try:
                # Clear GPU memory before processing each model
                clear_gpu_memory()

                # Preprocess image according to model
                if model_name == 'swincomer':
                    img_tensor = preprocess_for_swincomer(filepath)
                    latex = inference_swincomer(img_tensor)

                results[model_name] = {'latex': latex}
            except torch.cuda.OutOfMemoryError as e:
                print(f"CUDA out of memory with {model_name}: {str(e)}")
                results[model_name] = {'error': f'GPU out of memory. Try using CPU mode or one model at a time.'}
            except Exception as e:
                print(f"Error processing with {model_name}: {str(e)}")
                traceback.print_exc()
                results[model_name] = {'error': f'Processing error: {str(e)}'}

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

    # Clear GPU memory after processing all models
    clear_gpu_memory()

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
