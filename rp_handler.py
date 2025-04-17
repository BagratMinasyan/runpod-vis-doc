import runpod
from PIL import Image
import base64
import io
from inference import run_inference

def handler(event):
    input = event["input"]
    queries = input.get("queries")
    image_b64_list = input.get("images")

    images = []
    if image_b64_list:
        for b64_img in image_b64_list:
            img_data = base64.b64decode(b64_img)
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
            images.append(image)

    print(f"Running inference with {len(images)} image(s) and {len(queries or [])} query(ies)")
    results = run_inference(images=images if images else None, queries=queries if queries else None)

    response = {}
    if "image_embeddings" in results:
        response["image_embeddings"] = results["image_embeddings"].tolist()
    if "query_embeddings" in results:
        response["query_embeddings"] = results["query_embeddings"].tolist()

    return response

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
