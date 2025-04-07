import torch
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

def load_model():
    model = ColQwen2_5.from_pretrained(
        "vidore/colqwen2.5-v0.2",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).eval()
    processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.1")
    return model, processor
