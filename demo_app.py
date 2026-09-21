"""SwinCoMER demo: upload a handwritten expression image, get LaTeX back.

Run with `python demo_app.py`, then open http://localhost:5000.

The checkpoint is not in this repository. Point CHECKPOINT (or the
SWINCOMER_CHECKPOINT environment variable) at one before running, or the app
will still start and tell you what is missing on the page rather than crashing.
"""
import os
import sys
import time
import traceback
import uuid
import warnings

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms
from werkzeug.utils import secure_filename

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, "SwinCoMER"))

# PyTorch Lightning migrates the 1.x checkpoint format on load and is loud
# about it. The migration itself is fine; the warnings are not actionable.
warnings.filterwarnings("ignore", category=UserWarning, message=".*Multiple.*ModelCheckpoint.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Lightning automatically upgraded.*")

CHECKPOINT = os.environ.get(
    "SWINCOMER_CHECKPOINT",
    "SwinCoMER/checkpoints/ComerSwin-epoch=02-val_ExpRate=0.4550.ckpt",
)

# Overrides the checkpoint's own beam_size (8) because inference here is far
# more expensive than it should be: the encoder emits 6144 memory positions
# instead of 8x8=64, so every cross-attention step does ~96x the intended work
# (see the projection-layer note in SwinCoMER/comer/model/swin_encoder.py).
# Measured on CPU with the bundled example image: beam 8 = 407s, 4 = 212s,
# 2 = 107s, all three producing an identical result. Set to 0 to keep whatever
# the checkpoint was saved with.
BEAM_SIZE = int(os.environ.get("SWINCOMER_BEAM_SIZE", "2"))

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
MAX_UPLOAD_BYTES = 16 * 1024 * 1024

# ── Model import ────────────────────────────────────────────────────────────
# Imported at module level so an uninstalled package is reported once at
# startup instead of on every request. A failure here is recorded, not raised:
# the page must still load and explain what to install.
IMPORT_ERROR = None
try:
    from SwinCoMER.comer.datamodule import vocab
    from SwinCoMER.comer.lit_comer_swin import LitCoMER
except Exception as exc:  # ImportError, or a dependency blowing up on import
    LitCoMER = vocab = None
    IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    print(f"WARNING: could not import SwinCoMER — {IMPORT_ERROR}")
    print("Run: pip install -e ./SwinCoMER")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT} ({'found' if os.path.isfile(CHECKPOINT) else 'NOT FOUND'})")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Loaded lazily on the first recognition so startup stays fast and a missing
# checkpoint does not prevent the page from rendering its own diagnosis.
_model = None
_load_error = None


class ModelUnavailable(RuntimeError):
    """The model cannot serve, with the reason a user can act on."""


# ── Preprocessing ───────────────────────────────────────────────────────────
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


def preprocess(img_path):
    if not MATCH_TRAINING_PREPROCESSING:
        # Legacy path: ImageNet-normalized RGB.
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((256, 256), Image.LANCZOS)
        img_tensor = transforms.functional.to_tensor(img_resized)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return normalize(img_tensor).unsqueeze(0)

    # Mirror the training transform exactly, including cv2's INTER_LINEAR
    # (A.Resize's default) rather than PIL's LANCZOS.
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    arr = cv2.resize(arr, (256, 256), interpolation=cv2.INTER_LINEAR)

    img_tensor = torch.from_numpy(arr).float()            # [H, W] in [0, 255]
    img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
    return img_tensor.unsqueeze(0)                        # [1, 3, 256, 256]


# ── Model ───────────────────────────────────────────────────────────────────

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(use_cpu=False):
    """Restore the checkpoint once, then reuse it for every request."""
    global _model, _load_error

    if _model is not None:
        return _model

    if IMPORT_ERROR:
        raise ModelUnavailable(
            f"Chưa import được package 'comer' ({IMPORT_ERROR}). "
            "Chạy: pip install -e ./SwinCoMER"
        )
    if not os.path.isfile(CHECKPOINT):
        raise ModelUnavailable(
            f"Không tìm thấy checkpoint tại '{CHECKPOINT}'. "
            "Đặt biến môi trường SWINCOMER_CHECKPOINT trỏ tới file .ckpt."
        )

    target_device = torch.device("cpu") if use_cpu else DEVICE
    try:
        clear_gpu_memory()
        # NOTE: LitCoMER builds SwinV2PretrainedEncoder with pretrained=True,
        # so this downloads ImageNet weights and immediately overwrites them
        # with the checkpoint's. Harmless but slow, and it makes the first
        # start need network access.
        model = LitCoMER.load_from_checkpoint(CHECKPOINT, map_location=target_device)
        model.eval()
        try:
            model.to(target_device)
        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory moving the model to GPU — falling back to CPU.")
            model.to("cpu")
    except ModelUnavailable:
        raise
    except Exception as exc:
        _load_error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        raise ModelUnavailable(f"Nạp checkpoint thất bại: {_load_error}") from exc

    if BEAM_SIZE > 0:
        # approximate_joint_search reads beam_size off hparams, so overriding
        # it here is what actually takes effect at inference.
        model.hparams.beam_size = BEAM_SIZE

    _model = model
    _load_error = None
    print(f"Model loaded on {next(model.parameters()).device} "
          f"(beam_size={model.hparams.beam_size})")
    return _model


def recognize(img_path, use_cpu=False):
    """Returns (latex, score, elapsed_ms, device).

    An empty beam gives an empty latex string — a real outcome, not an error.
    The old demo returned the sentence "No result found" in the latex field,
    which the page then tried to typeset as if it were a formula.
    """
    model = load_model(use_cpu)
    started = time.perf_counter()

    img = preprocess(img_path)
    device = next(model.parameters()).device
    img = img.to(device)

    batch, _, height, width = img.shape
    mask = torch.zeros((batch, height, width), dtype=torch.bool, device=device)

    try:
        with torch.no_grad():
            hyps = model.approximate_joint_search(img, mask)
    finally:
        clear_gpu_memory()

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if not hyps:
        return "", 0.0, elapsed_ms, str(device)

    best = hyps[0]
    return vocab.indices2label(best.seq), float(best.score), elapsed_ms, str(device)


# ── Routes ──────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    """Everything needed to explain why recognition is or is not working.

    Reported before any image is sent, so a missing checkpoint shows up on
    page load instead of being discovered by a failed upload.
    """
    return jsonify({
        "import_ok": IMPORT_ERROR is None,
        "import_error": IMPORT_ERROR,
        "checkpoint": CHECKPOINT,
        "checkpoint_exists": os.path.isfile(CHECKPOINT),
        "loaded": _model is not None,
        "beam_size": _model.hparams.beam_size if _model else BEAM_SIZE,
        "device": str(next(_model.parameters()).device) if _model else None,
        "cuda_available": torch.cuda.is_available(),
        "last_error": _load_error,
    })


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Không có file nào được gửi"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Chưa chọn file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Chỉ chấp nhận ảnh PNG, JPG, BMP"}), 400

    use_cpu = request.form.get("use_cpu", "false").lower() == "true"

    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        latex, score, elapsed_ms, device = recognize(filepath, use_cpu)
    except ModelUnavailable as exc:
        # 503, not 500: a missing checkpoint is a setup state, not a crash.
        return jsonify({"error": str(exc)}), 503
    except torch.cuda.OutOfMemoryError:
        clear_gpu_memory()
        return jsonify({"error": "GPU hết bộ nhớ. Bật chế độ CPU rồi thử lại."}), 507
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Lỗi xử lý: {exc}"}), 500

    return jsonify({
        "latex": latex,
        "score": score,
        "elapsed_ms": elapsed_ms,
        "device": device,
        "filename": filename,
    })


if __name__ == "__main__":
    # debug=True enables the reloader, which imports this module twice and so
    # loads the checkpoint twice. Off by default; set FLASK_DEBUG=1 to opt in.
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
