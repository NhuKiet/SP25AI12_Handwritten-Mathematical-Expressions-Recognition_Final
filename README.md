# Mathematical Expression Recognition System

A comprehensive web application for recognizing mathematical expressions in images using multiple state-of-the-art deep learning models.

## Features

- **Multiple Model Support**:
  - CoMER (Convolutional and Multimodal Transformer)
  - SwinCoMER (Swin Transformer-based CoMER)
  - BTTR (Bidirectional Transformer)
  - PosFormer (Position-aware Transformer)
- **User-Friendly Interface**:
  - Easy image upload via drag-and-drop or file selection
  - Multiple model selection
  - Real-time processing status
  - Beautiful rendered math expressions using MathJax
- **Advanced Processing**:
  - GPU acceleration with CPU fallback
  - Automatic timeout protection
  - Memory optimization
  - Consistent image preprocessing (224x224 resolution)

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- Windows/Linux/MacOS

## Installation

1. **Clone the repository and its submodules**:
   ```bash
   git clone [your-repo-url]
   cd [your-repo-name]
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install model packages**:
   ```bash
   pip install -e ./CoMER
   pip install -e ./BTTR
   pip install -e ./PosFormer
   pip install -e ./SwinCoMER
   ```

4. **Verify directory structure**:
   ```
   ├── demo_app.py             # Main Flask application
   ├── templates/              # HTML templates
   │   └── index.html         # Web interface
   ├── uploads/               # Uploaded images directory
   ├── CoMER/                 # CoMER model
   ├── SwinCoMER/            # SwinCoMER model
   ├── BTTR/                 # BTTR model
   ├── PosFormer/            # PosFormer model
   ├── requirements.txt      # Python dependencies
   └── README.md            # This file
   ```

5. **Download model checkpoints**:
   - CoMER: Place in `CoMER/lightning_logs/version_0/checkpoints/`
   - SwinCoMER: Place in `SwinCoMER/checkpoints/`
   - BTTR: Place in `BTTR/checkpoints/`
   - PosFormer: Place in `PosFormer/lightning_logs/version_0/checkpoints/`

## Usage

1. **Start the application**:
   ```bash
   python demo_app.py
   ```

2. **Access the web interface**:
   - Open your browser
   - Go to `http://localhost:5000`

3. **Process images**:
   - Select desired models (multiple selection supported)
   - Upload an image containing mathematical expressions
   - Click "Process Image"
   - View results in LaTeX format and rendered mathematics

## Model Information

### CoMER
- Input: Grayscale images (224x224)
- Output: LaTeX expressions
- Processing: Fast, GPU-optimized

### SwinCoMER
- Input: RGB images (256x256)
- Output: LaTeX expressions
- Processing: Fast, GPU-optimized

### BTTR
- Input: Grayscale images (224x224)
- Output: LaTeX expressions
- Processing: Medium speed

### PosFormer
- Input: Grayscale images (224x224)
- Output: LaTeX expressions
- Processing: Can be slower, includes timeout protection

## Performance Optimization

- **GPU Usage**: Models automatically use GPU when available
- **Memory Management**: Automatic GPU memory cleanup
- **CPU Fallback**: Automatic fallback to CPU if GPU memory is insufficient
- **Timeout Protection**: 30-second timeout for each model to prevent hanging

## Troubleshooting

1. **GPU Out of Memory**:
   - Enable "Use CPU" option in the interface
   - Process with fewer models simultaneously
   - Try smaller images

2. **Slow Processing**:
   - Ensure GPU is available and CUDA is properly installed
   - Check system resources
   - Consider using faster models (CoMER/SwinCoMER)

3. **Model Not Available**:
   - Verify model checkpoints are in correct locations
   - Check model package installation
   - Verify CUDA/GPU availability

## License

[Your License Information]

## Acknowledgments

- CoMER, SwinCoMER, BTTR, and PosFormer model authors
- PyTorch and PyTorch Lightning teams
- Flask web framework 