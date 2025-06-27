from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available
from huggingface_hub import snapshot_download
import torch
import os
hf_token = os.getenv("RUNPOD_SECRET_hf_key")
print(hf_token)
def load_model():
    local_dir = snapshot_download(
        repo_id="Metric-AI/merged_models",
        repo_type="model",
        allow_patterns="exp18/*",
        local_dir="merged_model",
        token=hf_token  
    )

    model_path = f"{local_dir}/exp18/model"

    model = ColQwen2_5.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).eval()

    processor = ColQwen2_5_Processor.from_pretrained(model_path)
    return model, processor
