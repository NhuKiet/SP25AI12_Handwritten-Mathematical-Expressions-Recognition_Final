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
import threading
import queue

# Add all necessary module directories to path
current_dir = os.path.abspath('.')
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'BTTR'))
sys.path.append(os.path.join(current_dir, 'PosFormer'))
sys.path.append(os.path.join(current_dir, 'SwinCoMER'))
sys.path.append(os.path.join(current_dir, 'CoMER'))

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

try:
    from CoMER.comer.lit_comer import LitCoMER as PlainCoMER_LitCoMER_Class
    from CoMER.comer.datamodule import vocab as PlainCoMER_vocab_module
    AVAILABLE_MODELS['comer'] = True
    print("CoMER model (plain) is available")
except ImportError as e:
    AVAILABLE_MODELS['comer'] = False
    print(f"Warning: CoMER module (plain) not found or has errors: {str(e)}")

try:
    # Try different possible import paths for BTTR
    try:
        from BTTR.bttr.lit_bttr import LitBTTR
    except ImportError:
        # Alternative import path
        from bttr.lit_bttr import LitBTTR
    
    AVAILABLE_MODELS['bttr'] = True
    print("BTTR model is available")
except ImportError as e:
    AVAILABLE_MODELS['bttr'] = False
    print(f"Warning: BTTR module not found or has errors: {str(e)}")

try:
    # Try different possible import paths for PosFormer
    try:
        from PosFormer.Pos_Former.lit_posformer import LitPosFormer as PosFormer_LitPosFormer_Class
        from PosFormer.Pos_Former.datamodule import vocab as PosFormer_vocab_module
    except ImportError:
        # Alternative import path
        from Pos_Former.lit_posformer import LitPosFormer as PosFormer_LitPosFormer_Class
        from Pos_Former.datamodule import vocab as PosFormer_vocab_module
    
    AVAILABLE_MODELS['posformer'] = True
    print("PosFormer model is available")
except ImportError as e:
    AVAILABLE_MODELS['posformer'] = False
    print(f"Warning: PosFormer module not found or has errors: {str(e)}")

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
    },
    'comer': {
        'checkpoint': 'CoMER/lightning_logs/version_0/checkpoints/epoch=151-step=57151-val_ExpRate=0.6365.ckpt',
        'loaded': False,
        'model': None
    },
    'bttr': {
        'checkpoint': 'BTTR/checkpoints/epoch=197-step=69300-val_ExpRate=0.4477.ckpt',
        'loaded': False,
        'model': None
    },
    'posformer': {
        'checkpoint': 'PosFormer/lightning_logs/version_0/checkpoints/best.ckpt',
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
        elif model_name == 'comer':
            MODEL_CONFIGS[model_name]['model'] = PlainCoMER_LitCoMER_Class.load_from_checkpoint(
                MODEL_CONFIGS[model_name]['checkpoint'],
                max_length=10,
                map_location=target_device
            )
        elif model_name == 'bttr':
            MODEL_CONFIGS[model_name]['model'] = LitBTTR.load_from_checkpoint(
                MODEL_CONFIGS[model_name]['checkpoint'],
                max_length=10,
                map_location=target_device
            )
        elif model_name == 'posformer':
            print("Loading PosFormer model (you may see Lightning version migration warnings)")
            try:
                # Try loading on selected device
                MODEL_CONFIGS[model_name]['model'] = PosFormer_LitPosFormer_Class.load_from_checkpoint(
                    MODEL_CONFIGS[model_name]['checkpoint'],
                    max_length=10,
                    map_location=target_device
                )
            except torch.cuda.OutOfMemoryError:
                # If out of memory, force CPU
                print("CUDA out of memory when loading PosFormer. Loading on CPU instead.")
                MODEL_CONFIGS[model_name]['model'] = PosFormer_LitPosFormer_Class.load_from_checkpoint(
                    MODEL_CONFIGS[model_name]['checkpoint'], max_length=10,
                    map_location=torch.device('cpu')
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
def preprocess_for_swincomer(img_path):
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

def preprocess_for_comer(img_path):
    img = Image.open(img_path).convert("L")  # Convert to grayscale
    img_resized = img.resize((224, 224), Image.LANCZOS)
    img_tensor = transforms.functional.to_tensor(img_resized)
    # Note: No need for RGB normalization since we're using grayscale
    img_batched = img_tensor.unsqueeze(0)
    return img_batched

def preprocess_for_bttr(img_path):
    img = Image.open(img_path).convert("L")  # Convert to grayscale
    img_resized = img.resize((224, 224), Image.LANCZOS)
    img_tensor = transforms.ToTensor()(img_resized)
    if img_tensor.dim() == 2:
        img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension if needed
    return img_tensor

def preprocess_for_posformer(img_path):
    img = Image.open(img_path).convert("L")  # Convert to grayscale
    img_resized = img.resize((224, 224), Image.LANCZOS)
    img_tensor = transforms.ToTensor()(img_resized)
    if img_tensor.dim() == 2:
        img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension
    
    # Create image mask (assuming the same mask creation logic as CoMER)
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    img_mask = torch.zeros((H, W), dtype=torch.bool)
    
    # Add batch dimension for model input
    img_tensor = img_tensor.unsqueeze(0)
    img_mask = img_mask.unsqueeze(0)
    
    return img_tensor, img_mask

# Model inference functions
def inference_swincomer(img_tensor):
    model = MODEL_CONFIGS['swincomer']['model']
    img_tensor = img_tensor.to(device)
    B, _, H, W = img_tensor.shape
    img_mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)
    
    with torch.no_grad():
        hyps = model.approximate_joint_search(img_tensor, img_mask)
        if hyps:
            best_hyp = hyps[0]
            latex_str = SwinCoMER_vocab_module.indices2label(best_hyp.seq)
            return latex_str
        return "No result found"

def inference_comer(img_tensor):
    model = MODEL_CONFIGS['comer']['model']
    # Determine device model is on
    model_device = next(model.parameters()).device
    
    # Create a queue for the result
    result_queue = queue.Queue()
    
    def process_inference():
        try:
            # Clear memory before running inference
            clear_gpu_memory()
            
            with torch.no_grad():
                try:
                    # Try GPU first
                    if torch.cuda.is_available():
                        model.to('cuda')
                        img_tensor_gpu = img_tensor.to('cuda')
                        B, _, H, W = img_tensor_gpu.shape
                        img_mask = torch.zeros((B, H, W), dtype=torch.bool, device='cuda')
                        
                        hyps = model.approximate_joint_search(img_tensor_gpu, img_mask)
                        if hyps:
                            best_hyp = hyps[0]
                            latex_str = PlainCoMER_vocab_module.indices2label(best_hyp.seq)
                            result_queue.put(("success", latex_str))
                            return
                    else:
                        # If no GPU, use CPU
                        print("No GPU available, using CPU for CoMER")
                        model.to('cpu')
                        img_tensor_cpu = img_tensor.to('cpu')
                        B, _, H, W = img_tensor_cpu.shape
                        img_mask = torch.zeros((B, H, W), dtype=torch.bool, device='cpu')
                        
                        hyps = model.approximate_joint_search(img_tensor_cpu, img_mask)
                        if hyps:
                            best_hyp = hyps[0]
                            latex_str = PlainCoMER_vocab_module.indices2label(best_hyp.seq)
                            result_queue.put(("success", latex_str))
                            return
                    
                    result_queue.put(("error", "No result found"))
                except torch.cuda.OutOfMemoryError:
                    # If CUDA out of memory, fall back to CPU
                    print("CUDA out of memory during CoMER inference. Falling back to CPU...")
                    try:
                        model.to('cpu')
                        img_tensor_cpu = img_tensor.to('cpu')
                        B, _, H, W = img_tensor_cpu.shape
                        img_mask = torch.zeros((B, H, W), dtype=torch.bool, device='cpu')
                        clear_gpu_memory()  # Clear GPU memory after moving to CPU
                        
                        hyps = model.approximate_joint_search(img_tensor_cpu, img_mask)
                        if hyps:
                            best_hyp = hyps[0]
                            latex_str = PlainCoMER_vocab_module.indices2label(best_hyp.seq)
                            result_queue.put(("success", latex_str))
                        else:
                            result_queue.put(("error", "No result found"))
                    except Exception as e:
                        print(f"CPU fallback error: {str(e)}")
                        result_queue.put(("error", f"Error during CPU fallback: {str(e)}"))
                except Exception as e:
                    print(f"CoMER inference error: {str(e)}")
                    traceback.print_exc()
                    result_queue.put(("error", f"Error: {str(e)}"))
        except Exception as e:
            result_queue.put(("error", f"Error: {str(e)}"))
    
    # Start processing in a separate thread
    thread = threading.Thread(target=process_inference)
    thread.start()
    
    # Wait for result with timeout
    timeout = 30  # 30 seconds timeout
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            status, result = result_queue.get_nowait()
            if status == "success":
                return result
            else:
                return result
        except queue.Empty:
            time.sleep(0.1)  # Short sleep to prevent CPU overuse
    
    # If we get here, processing timed out
    return "Error: CoMER processing timed out. Try using a simpler image or another model."

def inference_bttr(img_tensor):
    model = MODEL_CONFIGS['bttr']['model']
    
    # Create a queue for the result
    result_queue = queue.Queue()
    
    def process_inference():
        try:
            # Clear memory before running inference
            clear_gpu_memory()
            
            with torch.no_grad():
                try:
                    # Try GPU first
                    if torch.cuda.is_available():
                        model.to('cuda')
                        # BTTR expects a 3D tensor [C,H,W], not 4D [B,C,H,W]
                        if img_tensor.dim() == 4:
                            img_tensor_gpu = img_tensor.squeeze(0).to('cuda')
                        else:
                            img_tensor_gpu = img_tensor.to('cuda')
                        
                        latex_str = model.beam_search(img_tensor_gpu, max_len=model.hparams.max_len)
                        if latex_str:
                            result_queue.put(("success", latex_str))
                            return
                    else:
                        # If no GPU, use CPU
                        print("No GPU available, using CPU for BTTR")
                        model.to('cpu')
                        if img_tensor.dim() == 4:
                            img_tensor_cpu = img_tensor.squeeze(0).to('cpu')
                        else:
                            img_tensor_cpu = img_tensor.to('cpu')
                        
                        latex_str = model.beam_search(img_tensor_cpu, max_len=model.hparams.max_len)
                        if latex_str:
                            result_queue.put(("success", latex_str))
                            return
                    
                    result_queue.put(("error", "No result found"))
                except torch.cuda.OutOfMemoryError:
                    # If CUDA out of memory, fall back to CPU
                    print("CUDA out of memory during BTTR inference. Falling back to CPU...")
                    try:
                        model.to('cpu')
                        if img_tensor.dim() == 4:
                            img_tensor_cpu = img_tensor.squeeze(0).to('cpu')
                        else:
                            img_tensor_cpu = img_tensor.to('cpu')
                        clear_gpu_memory()  # Clear GPU memory after moving to CPU
                        
                        latex_str = model.beam_search(img_tensor_cpu, max_len=model.hparams.max_len)
                        if latex_str:
                            result_queue.put(("success", latex_str))
                        else:
                            result_queue.put(("error", "No result found"))
                    except Exception as e:
                        print(f"CPU fallback error: {str(e)}")
                        result_queue.put(("error", f"Error during CPU fallback: {str(e)}"))
                except Exception as e:
                    print(f"BTTR inference error: {str(e)}")
                    traceback.print_exc()
                    result_queue.put(("error", f"Error: {str(e)}"))
        except Exception as e:
            result_queue.put(("error", f"Error: {str(e)}"))
    
    # Start processing in a separate thread
    thread = threading.Thread(target=process_inference)
    thread.start()
    
    # Wait for result with timeout
    timeout = 30  # 30 seconds timeout
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            status, result = result_queue.get_nowait()
            if status == "success":
                return result
            else:
                return result
        except queue.Empty:
            time.sleep(0.1)  # Short sleep to prevent CPU overuse
    
    # If we get here, processing timed out
    return "Error: BTTR processing timed out. Try using a simpler image or another model."

def inference_posformer(img_tensor, img_mask):
    model = MODEL_CONFIGS['posformer']['model']
    
    # Check where the model is located (GPU or CPU)
    model_device = next(model.parameters()).device
    
    # Move tensors to the same device as the model
    img_tensor = img_tensor.to(model_device)
    img_mask = img_mask.to(model_device)
    
    # Create a queue for the result
    result_queue = queue.Queue()
    
    def process_inference():
        try:
            # Clear memory before running inference
            clear_gpu_memory()
            
            with torch.no_grad():
                try:
                    # Try GPU first
                    if torch.cuda.is_available():
                        model.to('cuda')
                        img_tensor_gpu = img_tensor.to('cuda')
                        img_mask_gpu = img_mask.to('cuda')
                        
                        hyps = model.approximate_joint_search(img_tensor_gpu, img_mask_gpu)
                        if hyps:
                            best_hyp = hyps[0]
                            latex_str = PosFormer_vocab_module.indices2label(best_hyp.seq)
                            result_queue.put(("success", latex_str))
                            return
                    else:
                        # If no GPU, use CPU
                        print("No GPU available, using CPU for PosFormer")
                        model.to('cpu')
                        img_tensor_cpu = img_tensor.to('cpu')
                        img_mask_cpu = img_mask.to('cpu')
                        
                        hyps = model.approximate_joint_search(img_tensor_cpu, img_mask_cpu)
                        if hyps:
                            best_hyp = hyps[0]
                            latex_str = PosFormer_vocab_module.indices2label(best_hyp.seq)
                            result_queue.put(("success", latex_str))
                            return
                    
                    result_queue.put(("error", "No result found"))
                except torch.cuda.OutOfMemoryError:
                    # If CUDA out of memory, fall back to CPU
                    print("CUDA out of memory during PosFormer inference. Falling back to CPU...")
                    try:
                        model.to('cpu')
                        img_tensor_cpu = img_tensor.to('cpu')
                        img_mask_cpu = img_mask.to('cpu')
                        clear_gpu_memory()  # Clear GPU memory after moving to CPU
                        
                        hyps = model.approximate_joint_search(img_tensor_cpu, img_mask_cpu)
                        if hyps:
                            best_hyp = hyps[0]
                            latex_str = PosFormer_vocab_module.indices2label(best_hyp.seq)
                            result_queue.put(("success", latex_str))
                        else:
                            result_queue.put(("error", "No result found"))
                    except Exception as e:
                        print(f"CPU fallback error: {str(e)}")
                        result_queue.put(("error", f"Error during CPU fallback: {str(e)}"))
                except Exception as e:
                    print(f"PosFormer inference error: {str(e)}")
                    traceback.print_exc()
                    result_queue.put(("error", f"Error: {str(e)}"))
        except Exception as e:
            result_queue.put(("error", f"Error: {str(e)}"))
    
    # Start processing in a separate thread
    thread = threading.Thread(target=process_inference)
    thread.start()
    
    # Wait for result with timeout
    timeout = 30  # 30 seconds timeout
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            status, result = result_queue.get_nowait()
            if status == "success":
                return result
            else:
                return result
        except queue.Empty:
            time.sleep(0.1)  # Short sleep to prevent CPU overuse
    
    # If we get here, processing timed out
    return "Error: PosFormer processing timed out. Try using a simpler image or another model."

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
                elif model_name == 'comer':
                    img_tensor = preprocess_for_comer(filepath)
                    latex = inference_comer(img_tensor)
                elif model_name == 'bttr':
                    img_tensor = preprocess_for_bttr(filepath)
                    latex = inference_bttr(img_tensor)
                elif model_name == 'posformer':
                    img_tensor, img_mask = preprocess_for_posformer(filepath)
                    latex = inference_posformer(img_tensor, img_mask)
                
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