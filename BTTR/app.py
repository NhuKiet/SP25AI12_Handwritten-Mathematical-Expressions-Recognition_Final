from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from bttr.lit_bttr import LitBTTR

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model (done at startup to avoid reloading)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def load_model():
    global model
    ckpt = 'checkpoints/epoch=197-step=69300-val_ExpRate=0.4477.ckpt'
    model = LitBTTR.load_from_checkpoint(ckpt)
    model.eval()
    model.to(device)
    print(f"Model loaded successfully on {device}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(img_path):
    # Load image and convert to grayscale
    img = Image.open(img_path).convert("L")
    img = ToTensor()(img)
    
    # Ensure image has shape [1, H, W]
    if img.dim() == 2:
        img = img.unsqueeze(0)
    elif img.shape[0] != 1:
        raise ValueError(f"Unexpected image shape {img.shape}, expected [1, H, W]")
    
    # Move to device and run model
    img = img.to(device)
    with torch.no_grad():
        latex = model.beam_search(img)
    
    return latex

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            latex = process_image(filepath)
            return jsonify({'latex': latex})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    load_model()
    app.run(debug=True)