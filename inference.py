import torch
from PIL import Image
from model_loader import load_model

model, processor = load_model()

def run_inference(images: list[Image.Image] = None, queries: list[str] = None):
    device = model.device
    results = {}

    if images:
        batch_images = processor.process_images(images).to(device)
        with torch.no_grad():
            image_embeddings = model(**batch_images)
        results["image_embeddings"] = image_embeddings.cpu()

    if queries:
        batch_queries = processor.process_queries(queries).to(device)
        with torch.no_grad():
            query_embeddings = model(**batch_queries)
        results["query_embeddings"] = query_embeddings.cpu()

    return results