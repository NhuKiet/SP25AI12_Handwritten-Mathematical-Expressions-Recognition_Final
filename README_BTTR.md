# Using BTTR for Mathematical Expression Recognition

This guide explains how to set up and use the BTTR (Bidirectional Transformer for Handwritten Mathematical Expression Recognition) model for recognizing mathematical expressions in images.

## Setup

1. **Install BTTR Package**

   Run the installation helper script:
   ```
   python install_bttr.py
   ```
   This will install the BTTR package and its dependencies.

2. **Download Model Checkpoint**

   The BTTR model requires a pre-trained checkpoint. You have two options:
   
   - **Option 1**: Download from the official repository and place it at:
     ```
     BTTR/lightning_logs/version_0/checkpoints/epoch=259-step=97759.ckpt
     ```
   
   - **Option 2**: Train your own model by following the instructions in the BTTR repository.

## Usage

### Using bttr_test.py

The `bttr_test.py` script provides a simple way to test the BTTR model on image files:

```
python bttr_test.py --image your_image.jpg
```

Options:
- `--image`: Path to input image (default: testdoan.jpeg)
- `--checkpoint`: Path to model checkpoint (will search common locations if not specified)
- `--mode`: Image preprocessing mode (choices: minimal, standard, aggressive; default: aggressive)
- `--cpu`: Force CPU inference even if GPU is available

Example:
```
python bttr_test.py --image testdoan.jpeg --mode aggressive
```

### Using app.py

The `app.py` script has been modified to use BTTR instead of CoMER:

```
python app.py
```

## Preprocessing Modes

- **minimal**: Just load and convert to grayscale tensor (closest to training)
- **standard**: Basic processing with CLAHE enhancement
- **aggressive**: Grayscale + threshold optimization (best for handwritten expressions)

## Troubleshooting

1. **Checkpoint Not Found**
   - Ensure you've downloaded the checkpoint to the correct location
   - Specify the checkpoint path with `--checkpoint` argument

2. **Import Errors**
   - Run `python install_bttr.py` to install dependencies
   - Ensure PyTorch and torchvision are installed

3. **CUDA Errors**
   - Try using CPU mode with `--cpu` flag
   - Check GPU drivers and CUDA installation

## References

- [Official BTTR Repository](https://github.com/Green-Wood/BTTR)
- [Paper: Bidirectional Transformer for Handwritten Mathematical Expression Recognition](https://arxiv.org/abs/2105.02412) 