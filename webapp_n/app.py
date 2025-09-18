from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import io
from .utils import load_generator, generate_image
from PIL import Image

app = FastAPI()

# Load model once at startup
MODEL_PATH = "./checkpoints_n/i2s_v3/latest_net_G.pth"
DEVICE = "cpu"  # App Service doesn't provide GPU
netG = load_generator(MODEL_PATH, DEVICE)

@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    # Read file into memory
    contents = await file.read()
    input_image = Image.open(io.BytesIO(contents)).convert("RGB")

    img = generate_image(netG, input_image, DEVICE)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
