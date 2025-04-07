import runpod
from PIL import Image
import base64
import io
from .inference import run_inference

def handler(event):
    input = event["input"]
    queries = input.get("queries", [])
    image_b64_list = input.get("images", [])  # list of base64 encoded strings

    images = []
    for b64_img in image_b64_list:
        img_data = base64.b64decode(b64_img)
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        images.append(image)

    print(f"Running inference with {len(images)} image(s) and {len(queries)} query(ies)")
    results = run_inference(images, queries)

    return {"scores": results}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
