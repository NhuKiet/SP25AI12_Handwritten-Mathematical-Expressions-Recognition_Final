# Math Expression Recognition Demo App

This demo application allows you to test multiple mathematical expression recognition models on your own images.

## Features

- Upload images containing mathematical expressions
- Select between three state-of-the-art models:
  - CoMER (Convolutional and Multimodal Transformer for Optical Mathematical Expression Recognition)
  - BTTR (Bidirectional Transformer for Optical Mathematical Expression Recognition)
  - PosFormer (Position-aware Transformer for Optical Mathematical Expression Recognition)
- View LaTeX output from each model
- Rendered expressions using MathJax

## Setup and Installation

1. Make sure you have all the dependencies installed:
   ```
   pip install flask torch torchvision pillow opencv-python pytorch-lightning
   ```

2. Make sure the model repositories are properly installed:
   - All three models (CoMER, BTTR, and PosFormer) should be in the project root directory
   - If necessary, install the modules in development mode:
     ```
     pip install -e ./CoMER
     pip install -e ./BTTR
     pip install -e ./PosFormer
     ```

3. Make sure the model checkpoints are available in the following locations:
   - CoMER: `CoMER/checkpoints/ComerSwin-epoch=02-val_ExpRate=0.4550.ckpt`
   - BTTR: `BTTR/checkpoints/epoch=197-step=69300-val_ExpRate=0.4477.ckpt`
   - PosFormer: `PosFormer/lightning_logs/version_0/checkpoints/best.ckpt`

4. Verify the directory structure:
   ```
   ├── demo_app.py             # Main Flask application
   ├── templates/              # Templates directory
   │   └── index.html          # Web interface
   ├── uploads/                # Directory for uploaded images
   ├── CoMER/                  # CoMER model directory
   ├── BTTR/                   # BTTR model directory
   └── PosFormer/              # PosFormer model directory
   ```

5. Run the app:
   ```
   python demo_app.py
   ```

6. Open your browser and go to: `http://127.0.0.1:5000`

## Usage

1. Select one or more models to use
2. Upload an image containing a mathematical expression (JPG, PNG, or BMP)
3. Click "Process Image"
4. View the LaTeX output and rendered expressions from each model

## Troubleshooting

If you encounter import errors:

1. Make sure the model directories (CoMER, BTTR, PosFormer) are in the correct locations
2. Try installing the packages in development mode:
   ```
   pip install -e ./CoMER
   pip install -e ./BTTR
   pip install -e ./PosFormer
   ```
3. Check the model directory structure to ensure it matches what the imports expect

If you encounter model loading errors:

1. Verify that all dependencies are installed correctly
2. Check that the model checkpoint files exist in the correct locations
3. Examine the console output for specific error messages

If you encounter memory issues:

1. Try using smaller images
2. Run only one model at a time
3. If using GPU, free up GPU memory or switch to CPU mode

Other tips:

- The first processing may take some time as the models are loaded into memory
- GPU acceleration is used if available, otherwise CPU will be used
- Images are preprocessed specifically for each model to achieve the best results

## File Structure

- `demo_app.py` - The main Flask application
- `templates/index.html` - The web interface
- `uploads/` - Directory for uploaded images

## Troubleshooting

If you encounter any issues:

1. Check that all dependencies are installed correctly
2. Verify that the model checkpoint files exist in the correct locations
3. Check for any errors in the console output
4. Try using smaller images if you encounter memory issues 