# load_checkpoint_example.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CKPT_DIR = "checkpoints/deepseek_ocr_lora_checkpoint_1000"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    CKPT_DIR,
    trust_remote_code=True,
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    CKPT_DIR,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).eval()

print("✅ Model loaded!")

# ---- Example inference ----
prompt = "识别此图像中的文字：<image>"

inputs = tokenizer(prompt, return_tensors="pt")
print("Tokens:", inputs["input_ids"].shape)

print("⚠️ Demo: 由于没有输入 image，这里只展示 tokenizer 是否正常工作")
