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
- Git LFS (for handling large files)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/[your-username]/[repo-name].git
   cd [repo-name]
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model checkpoints**:
   
   The model checkpoints are large files and are not included in the repository. Download them from the following locations:

   - **CoMER**:
     - File: `epoch=151-step=57151-val_ExpRate=0.6365.ckpt`
     - Size: ~50MB
     - Place in: `CoMER/lightning_logs/version_0/checkpoints/`
     - [Download Link]

   - **SwinCoMER**:
     - File: `ComerSwin-epoch=02-val_ExpRate=0.4550.ckpt`
     - Place in: `SwinCoMER/checkpoints/`
     - [Download Link]

   - **BTTR**:
     - File: `epoch=197-step=69300-val_ExpRate=0.4477.ckpt`
     - Place in: `BTTR/checkpoints/`
     - [Download Link]

   - **PosFormer**:
     - File: `best.ckpt`
     - Place in: `PosFormer/lightning_logs/version_0/checkpoints/`
     - [Download Link]

4. **Install model packages**:
   ```bash
   pip install -e ./CoMER
   pip install -e ./BTTR
   pip install -e ./PosFormer
   pip install -e ./SwinCoMER
   ```

5. **Verify directory structure**:
   ```
   ├── demo_app.py             # Main Flask application
   ├── templates/              # HTML templates
   │   └── index.html         # Web interface
   ├── uploads/               # Uploaded images directory (created automatically)
   ├── CoMER/                 # CoMER model
   ├── SwinCoMER/            # SwinCoMER model
   ├── BTTR/                 # BTTR model
   ├── PosFormer/            # PosFormer model
   ├── requirements.txt      # Python dependencies
   ├── .gitignore           # Git ignore rules
   └── README.md            # This file
   ```

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

## Development Notes

### Repository Management

1. **Large Files**:
   - Model checkpoints are not included in the repository
   - Download them separately from the provided links
   - The `.gitignore` file is configured to exclude large binary files

2. **Virtual Environment**:
   - It's recommended to use a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # Linux/Mac
     venv\Scripts\activate     # Windows
     ```

3. **Contributing**:
   - Fork the repository
   - Create a feature branch
   - Do not commit model checkpoints or large binary files
   - Submit pull requests for code changes only

### Common Issues

1. **Git Push Errors**:
   - If you get timeout errors while pushing, check for large files
   - Use `git status` to verify no checkpoint files are being tracked
   - Use `git clean -fd` to remove untracked files (careful!) 