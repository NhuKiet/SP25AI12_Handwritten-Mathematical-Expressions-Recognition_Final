from bttr.lit_bttr import LitBTTR
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize


# Load the image and convert to grayscale
img_path = 'ISICal19_1201_em_763.bmp'

img = Image.open(img_path).convert("L")  # Convert to grayscale (C=1)
img = ToTensor()(img)  # Convert to tensor [1, H, W] hoặc [H, W]
# Debugging - Kiểm tra kích thước ảnh
print("Image shape after ToTensor:", img.shape)

# Đảm bảo ảnh có đúng dạng [1, H, W]
if img.dim() == 2:  
    img = img.unsqueeze(0)  # Thêm kênh màu nếu thiếu (C=1)
elif img.shape[0] != 1:  
    raise ValueError(f"Unexpected image shape {img.shape}, expected [1, H, W]")

# Load the model
ckpt = 'checkpoints/epoch=197-step=69300-val_ExpRate=0.4477.ckpt'
model = LitBTTR.load_from_checkpoint(ckpt)
model.eval()

# Ensure device compatibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
img = img.to(device)

# **Truyền ảnh vào beam_search mà không squeeze thêm**
hyp = model.beam_search(img)  

print("Recognized LaTeX:", hyp)
