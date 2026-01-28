from transformers import AutoModel, AutoTokenizer
import torch
import os


os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

model_name = os.environ.get("MODEL_PATH", "deepseek-ai/DeepSeek-OCR-2")
attn_impl = os.environ.get("ATTN_IMPL", "sdpa")  # set to flash_attention_2 if you installed flash-attn


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation=attn_impl,
    trust_remote_code=True,
    use_safetensors=True,
)
model = model.eval().cuda().to(torch.bfloat16)



# prompt = "<image>\nFree OCR. "
prompt = os.environ.get("PROMPT") or "<image>\n<|grounding|>Convert the document to markdown. "
image_file = os.environ.get("IMAGE_FILE", "your/image/path")
output_path = os.environ.get("OUTPUT_PATH", "your/output/path")




res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 768, crop_mode=True, save_results = True)
