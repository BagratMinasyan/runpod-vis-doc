import torch
from PIL import Image
from model_loader import load_model

model, processor = load_model()

def run_inference(images: list[Image.Image], queries: list[str]):
    device = model.device

    batch_images = processor.process_images(images).to(device)
    batch_queries = processor.process_queries(queries).to(device)

    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    #return processor.score_multi_vector(query_embeddings, image_embeddings).tolist()
    return {
        "image_embeddings": image_embeddings.cpu(),
        "query_embeddings": query_embeddings.cpu()
    }
