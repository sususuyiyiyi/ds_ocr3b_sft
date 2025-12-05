import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

CKPT_DIR = "checkpoints/deepseek_ocr_lora_checkpoint_1000"

tokenizer = AutoTokenizer.from_pretrained(
    CKPT_DIR,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    CKPT_DIR,
    trust_remote_code=True,
    torch_dtype=torch.float16,
).eval()

print("✅ Model & tokenizer loaded from", CKPT_DIR)
