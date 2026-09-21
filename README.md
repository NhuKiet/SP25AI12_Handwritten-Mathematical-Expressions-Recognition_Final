# Handwritten Mathematical Expressions Recognition (SwinCoMER)

**Capstone Project SP25AI12** — *Nhận dạng biểu thức toán học viết tay*

An end-to-end system that converts images of handwritten mathematical expressions into machine-readable LaTeX. The recognition model is **Swin Transformer-CoMER**, a hybrid architecture proposed by this project that pairs a Swin Transformer V2 encoder with CoMER's coverage-aware Transformer decoder. A Flask web application wraps the model so an image can be uploaded and the result viewed as both raw LaTeX and rendered mathematics.

📄 **Full project report**: [SP25AI12 — Handwritten Mathematical Expressions Recognition (Final Document)](docs/SP25AI12_Handwritten%20Mathematical%20Expressions%20Recognition_FinalDocument.docx)

> This repository previously also contained BTTR, CoMER and PosFormer, which were implemented as comparison baselines during the research phase. They have been removed; only the proposed SwinCoMER model remains. Their benchmark numbers are kept in the [Results](#results) table below for reference.

## Project Information

| | |
|---|---|
| **Project code** | SP25AI12 |
| **Duration** | 01/2025 – 05/2025 |
| **Supervisors** | Dr. Dang Ngoc Minh Duc, Dr. Nguyen Tuan Cuong |

**Team**

| Member | Role |
|---|---|
| Bui Nhu Kiet (SE170342) — Leader | AI Engineer, Project Manager, Full-stack Software Engineer |
| Truong Tu Kha (SE160885) | Researcher, Deployment |
| Pham Ngoc Hoang Anh (SE160891) | Model Developer, Evaluation |

## Motivation

Mathematical expressions appear constantly in teaching, engineering and scientific writing, but transcribing them by hand into digital form is slow and error-prone. The difficulty is not just character recognition: expressions are inherently **two-dimensional** (fractions, superscripts, subscripts, nested radicals, matrices), symbol sets are large, and handwriting varies enormously — `9` versus `g`, or `x` versus `×`, are routinely confused.

Automating this transcription helps students and teachers digitize notes, lets researchers move handwritten derivations into publishable form, and makes mathematical content accessible to visually impaired readers through machine-readable output.

## The Model

### Why Swin Transformer + CoMER

Transformer encoder–decoder models improved HMER substantially, but plain Transformers suffer from a **lack of coverage** — the decoder has no memory of which image regions it has already attended to, so it skips symbols or emits the same one twice. CoMER solved this with an **Attention Refinement Module (ARM)**, which reuses past alignment information (self-coverage within a layer, cross-coverage across layers) to reweight attention, without breaking the Transformer's parallelism.

CoMER's weakness lay elsewhere: its **DenseNet CNN encoder** captures local features well but has limited global context, making it hard to relate, say, an integral's upper limit to its integrand across a wide image.

This project replaces that encoder with a **Swin Transformer V2**. Shifted Window Attention gives long-range dependency modelling at cost linear in image size, and the hierarchical patch-merging stages produce multi-scale features — fine strokes preserved early, overall layout encoded late. The result marries a richer 2D image representation with coverage-aware sequence generation.

### Architecture

**Encoder — Swin Transformer V2**
- Input partitioned into non-overlapping 4×4 patches, each linearly projected to a 48-dim embedding
- Four stages of Swin blocks (2, 2, 6, 2 layers) interleaved with patch-merging, halving resolution and doubling channels at each step
- A 1×1 convolution + BatchNorm + ReLU projects final feature maps to the decoder's model dimension
- A learnable 2D positional encoding is added to each flattened patch embedding

**Decoder — CoMER-style Transformer with ARM**
- Masked self-attention for autoregressive generation
- **ARM** consumes the attention map plus accumulated coverage history and reweights attention to avoid both under-attending (missed symbols) and over-attending (duplicates)
- Multi-head cross-attention to encoder features, then a two-layer feed-forward network; residual connections and layer norm throughout

**Inference — beam search**
- `approximate_joint_search` with bidirectional beam search; length normalization counteracts bias toward short sequences and a mild coverage penalty discourages repeated attention
- Highest-scoring complete hypothesis is returned

Roughly **54M parameters** for the tiny variant — larger and slower than DenseNet-based CoMER, in exchange for accuracy on complex layouts.

> **ARM was disabled in the run that produced the shipped checkpoint.** `train.py` passed `cross_coverage=False, self_coverage=False`, and `_build_transformer_decoder` only constructs the Attention Refinement Module when at least one of those is set — so the trained model is a Swin encoder with a plain Transformer decoder, with no coverage mechanism. `config.yaml` sets both to `true`, but `train.py` never reads that file. Both flags now default to `True`, so future runs include ARM. Existing checkpoints carry the old values in their saved hyperparameters and still load exactly as trained. The benchmark numbers below were produced **without** ARM.

## Dataset

Training used a **Combined Dataset** merging a custom corpus with the CROHME benchmarks.

**Custom dataset** — built from **MathWriting 2024**. InkML stroke recordings were normalized so the bounding box's longest side fits a unit square, ultra-short strokes discarded, and trajectories smoothed with a Savitzky–Golay filter (window 5, polynomial order 2). Cleaned strokes were rendered to PNG via Cairo, and each file's `normalizedLabel` field extracted into an image→LaTeX CSV. Split 90/5/5.

**CROHME 2014 / 2016 / 2019** — standard HMER benchmarks sharing one 8,834-sample training set, with separate test sets.

| Source | Train | Validation | Test |
|---|---|---|---|
| Custom (MathWriting 2024) | 206,877 | 11,493 | 11,494 |
| CROHME 2014 | 8,834 | 983 | 983 |
| CROHME 2016 | 8,834 | — | 1,147 |
| CROHME 2019 | 8,834 | — | 1,199 |
| **Combined** | **215,711** | **12,476** | **14,823** |

### Preprocessing

- **Images**: converted to RGB, resized to 256×256, pixel values normalized. Grayscale inputs are repeated to three channels for RGB compatibility. Corrupted images are replaced with a blank 256×256 RGB image to keep the dataset intact.
- **Augmentation (training only)**: solarization at 50% probability to simulate lighting variation, and erosion or dilation at 50% probability to mimic stroke-width variation. Validation and test sets get resizing only, to preserve authenticity.
- **LaTeX**: tokenized into commands, symbols and variables (`\frac`, `x`, `2`, …); a unified vocabulary is built over the combined training set with padding, sequence-boundary and unknown tokens. Malformed entries are excluded and logged.

### Training configuration

| | |
|---|---|
| Hardware | 2× NVIDIA T4 (16 GiB each) |
| Loss | Cross-entropy with label smoothing |
| Optimizer | AdamW with warm-up |
| Learning rate | 0.001 |
| Batch size | 6 |
| Max epochs | 200, with early stopping on validation-loss plateau |
| Tracking | Weights & Biases |

The values above are from the project report. `train.py` — the script that actually runs — differs: batch size 16, learning rate 0.0001, a single GPU, and plain `Adam` with `weight_decay` (L2 inside Adam, not AdamW) on a `ReduceLROnPlateau` schedule with no warm-up. `config.yaml` is a PyTorch Lightning CLI config that nothing reads, and its values differ again. **`train.py` is the source of truth.**

## Results

Expression Recognition Rate (**ExpRate**) — the percentage of test expressions whose LaTeX output matches the ground truth *exactly*. A single symbol error makes the expression wrong.

| Model | Train set | CROHME 2014 | CROHME 2016 | CROHME 2019 |
|---|---|---|---|---|
| BTTR | CROHME | 53.96% | 52.31% | 52.96% |
| CoMER | CROHME | 59.33% | 59.81% | 62.97% |
| PosFormer | CROHME | 62.68% | 61.03% | 64.97% |
| **SwinCoMER** (this project) | Combined | **51.41%** | **45.24%** | **50.37%** |

BTTR results are without scale augmentation; the other three include it.

Alongside ExpRate, the project reports **≤1 / ≤2 / ≤3 error** rates — the share of expressions within that Levenshtein distance of the ground truth — since a result needing one manual correction is still useful in practice.

**These numbers do not tell the whole story.** SwinCoMER trails PosFormer and CoMER on CROHME, but the comparison is not like-for-like: the baselines were trained on CROHME's 8,834 samples and evaluated on CROHME test sets, while SwinCoMER was trained on the 215,711-sample Combined Dataset and is therefore being measured **out of domain** here.

That difference shows up clearly in practice. On real handwriting that appears in no test set, SwinCoMER generalizes markedly better than the three baselines, which recognize CROHME-style input well but degrade sharply outside it. A benchmark ExpRate rewards fitting the benchmark's distribution; it does not measure the robustness that matters when a student photographs their own notes. The CROHME gap is largely the cost of not specializing to CROHME.

Error analysis found the dominant remaining failure mode to be visually similar symbols (`9` vs `g`) under noisy or irregular handwriting.

### A caveat on the checkpoint filename

The checkpoint shipped with the demo is named `ComerSwin-epoch=02-val_ExpRate=0.4550.ckpt`, but **that 0.4550 was never an ExpRate**. The original `validation_step` computed sequence accuracy under *teacher forcing* — the decoder receives the ground-truth prefix at every step — and logged it as `val_ExpRate`. Teacher-forced accuracy is substantially more optimistic than autoregressive decoding, and `ModelCheckpoint` was selecting on it.

This has been corrected. The teacher-forced figure is now logged as `val_token_seq_acc`, and `val_ExpRate` runs a real beam search, matching the procedure used at test time. Checkpoints produced by future runs are selected on the real metric; the filename above is a historical artifact of the old behaviour.

## Getting Started

### Prerequisites

- Python 3.9+
- PyTorch 1.12+ with CUDA (GPU strongly recommended; the app falls back to CPU)
- 16GB RAM
- Windows 11, macOS Ventura+, or Ubuntu 20.04 LTS+

Reference local setup from the report: NVIDIA RTX 3050 Ti Laptop GPU, 16GB RAM.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SP25AI12/CapstoneProject_SP25AI12.git
   cd CapstoneProject_SP25AI12
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the model package**:
   ```bash
   pip install -e ./SwinCoMER
   ```

5. **Add the model checkpoint**:

   Checkpoints are **not** stored in this repository — `.gitignore` excludes `*.ckpt`, `*.pth`, `checkpoints/` and `lightning_logs/`. Obtain the trained weights separately and place them at the exact path the app expects:

   ```
   SwinCoMER/checkpoints/ComerSwin-epoch=02-val_ExpRate=0.4550.ckpt
   ```

   To use a different checkpoint, edit the `checkpoint` path in `MODEL_CONFIGS` near the top of `demo_app.py`.

### Running the app

```bash
python demo_app.py
```

Open `http://localhost:5000`, upload an image (`.png`, `.jpg`, `.jpeg`, `.bmp`, max 16MB) and click **Process Image**. The result appears as a LaTeX string and as mathematics rendered by MathJax.

The model is loaded lazily on the first request, so the first recognition takes noticeably longer than later ones.

### Preprocessing mode

`demo_app.py` exposes a `MATCH_TRAINING_PREPROCESSING` flag.

The training pipeline feeds the model grayscale images with values in **[0, 255]** — Albumentations' `ToTensorV2` does not divide by 255, and no `A.Normalize` appears anywhere in `transforms.py`. The demo originally fed ImageNet-normalized RGB instead, which the model never saw during training. Swin V2's LayerNorm after patch embedding absorbs most of a uniform affine rescale, so this skew is not catastrophic, but it is still a mismatch.

With the flag `True` (default) the demo mirrors the training transform exactly. Set it to `False` to restore the previous behaviour. Worth A/B-testing on your own samples before committing to either.

## How It Works

```
image upload → preprocessing → encoder → decoder → post-processing → rendered output
```

1. **Upload** — the Flask app saves the image to `uploads/` under a UUID-prefixed filename
2. **Preprocessing** — converted to RGB, resized to 256×256, normalized with ImageNet mean/std, batched into a `[1, 3, 256, 256]` tensor
3. **Inference** — the Swin encoder produces hierarchical feature maps; the ARM-equipped decoder generates token IDs autoregressively via beam search
4. **Post-processing** — token IDs are mapped back to LaTeX symbols through the vocabulary (`frac → \frac`) and assembled into a valid LaTeX string
5. **Presentation** — the string is returned as JSON and rendered in-browser by MathJax

## Repository Structure

```
├── demo_app.py            # Flask application (upload, inference, JSON results)
├── templates/
│   └── index.html         # Web interface with MathJax rendering
├── uploads/               # Uploaded images (created automatically)
├── docs/                  # Project documentation
│   └── SP25AI12_Handwritten Mathematical Expressions Recognition_FinalDocument.docx
├── SwinCoMER/             # SwinCoMER model package
│   ├── comer/             # Model definition (encoder, decoder, datamodule)
│   ├── checkpoints/       # Place the .ckpt here (not tracked by git)
│   ├── config.yaml        # Training hyperparameters
│   ├── train.py           # Training entrypoint
│   ├── Test.py            # Evaluation entrypoint
│   ├── convert2symLG/     # CROHME tool: LaTeX → symLG conversion
│   ├── lgeval/            # CROHME tool: symLG comparison
│   └── example/           # Sample image + demo notebook
├── fix_numpy.py           # Helper: downgrade NumPy if PyTorch complains
├── requirements.txt
├── .gitignore
└── README.md
```

## Training and Evaluation

Training is separate from the web app. Both entrypoints expect a CROHME-format `data.zip` that is **not** included in this repository, and their paths are hardcoded to the environments they were last run in (Google Colab and Kaggle respectively) — edit those paths before running.

```bash
cd SwinCoMER
python train.py           # trains, logs to Weights & Biases
python Test.py            # evaluates a checkpoint on the test set
```

For the official CROHME metrics, use the bundled tools in `convert2symLG/` and `lgeval/` via `eval_all.sh` (requires Perl 5).

## Troubleshooting

1. **"SwinCoMER module not found or has errors"** on startup
   - Run `pip install -e ./SwinCoMER`
   - Read the traceback printed to the console — a missing dependency surfaces here as an import error

2. **"Failed to load model: swincomer"**
   - Verify the checkpoint exists at the exact path in installation step 5
   - Confirm the filename matches `MODEL_CONFIGS` in `demo_app.py`

3. **GPU out of memory**
   - Enable the "Use CPU" switch in the interface
   - Try a smaller image

4. **NumPy / PyTorch compatibility errors**
   - Run `python fix_numpy.py`, which downgrades NumPy to 1.24.3

5. **Slow processing**
   - Confirm CUDA is detected — the console prints `Using device: cuda` or `cpu` at startup

## Known Limitations

- **Handwriting variability** — illegible or highly ambiguous handwriting causes errors, especially between visually similar symbols
- **Complex structures** — multi-line equations, intricate matrices and rare notation (e.g. logic symbols like ⊨, ⊢) are underrepresented in training data and often misrecognized
- **Computational cost** — the Transformer encoder needs a capable GPU; real-time use on mobile or low-power devices is not currently feasible
- **Symbol scope** — coverage targets common high-school and undergraduate notation, not region- or domain-specific scripts

## Future Work

- Integrate symbolic structure parsing and language models (BERT/GPT) to resolve ambiguous symbols using semantic context
- Quantization or pruning for lightweight mobile deployment
- Export to ONNX or TorchScript to cut inference latency
- Expand the dataset through synthetic generation (e.g. GC-DDPM) and crowdsourcing to close diversity gaps
- Broaden support for multilingual and domain-specific notation

## Development Notes

- **Never commit checkpoints or datasets.** `.gitignore` already excludes them; run `git status` before pushing to confirm no `.ckpt` files are staged.
- `demo_app.py` runs with `debug=True` — fine for local demos, unsuitable for production.
- Uploaded images accumulate in `uploads/` and are never cleaned up automatically.

## Acknowledgments

- **CoMER** — Zhao & Gao, [*Modeling Coverage for Transformer-based Handwritten Mathematical Expression Recognition*](https://arxiv.org/abs/2207.04410)
- **BTTR** and **PosFormer** authors, whose models served as comparison baselines
- The CROHME organizers for the benchmark datasets and evaluation tools
- Google Research for the MathWriting 2024 dataset
- PyTorch, PyTorch Lightning, Flask and MathJax

## License

[Your License Information]
