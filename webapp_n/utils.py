import sys
import os
import torch
import torchvision.transforms as transforms
from PIL import Image

# Make sure we can import pix2pix repo code (models/network)
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../pix2pix_code_n"))
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from models.networks import define_G  # import generator architecture


def load_generator(model_path, device="cpu"):
    """
    Load trained generator from checkpoint.
    """
    netG = define_G(
        input_nc=3,
        output_nc=3,
        ngf=64,
        netG="unet_256",
        norm="batch",
        use_dropout=False,
    )
    checkpoint = torch.load(model_path, map_location=device)
    netG.load_state_dict(checkpoint)
    netG.to(device)
    netG.eval()
    return netG


def generate_image(model, image_input, device="cpu"):
    """
    Run inference on a local image file using the trained generator.
    Returns a PIL image.
    """
    # Load image (accept both path or PIL.Image)
    if isinstance(image_input, str):  # path
        img = Image.open(image_input).convert("RGB")
    else:  # assume PIL.Image
        img = image_input.convert("RGB")

    # preprocess
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # inference
    with torch.no_grad():
        fake_B = model(input_tensor)

    # postprocess
    fake_B = (fake_B.squeeze().cpu() * 0.5 + 0.5).clamp(0, 1)
    fake_img = transforms.ToPILImage()(fake_B)

    return fake_img
