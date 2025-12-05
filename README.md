# 📘 DeepSeek-OCR LoRA Fine-tuning (Chinese OCR)

A complete, reproducible pipeline for fine-tuning DeepSeek-OCR-3B using LoRA, Unsloth, and HuggingFace.
This project focuses on improving real-world Chinese OCR accuracy (CER), and provides a fully working multi-modal training & evaluation framework.

## 🔥 1. Motivation

DeepSeek-OCR delivers strong zero-shot OCR, but in many business scenarios (bills, receipts, medical records, screenshots) baseline accuracy is unstable:

Frequent hallucination ("请输入...", "图片内容是...")

Character order errors

Over-generated text

Shape-similar character confusion (“瘦/受”, “点/典”)

High baseline CER (≈1.0)

Goal：Build a LoRA fine-tuning pipeline that can significantly improve OCR performance on domain data—
and make the whole process reproducible, interrupt-resistant, and suitable for long training sessions.

## 🚀 2. Project Highlights
✔ Full multi-modal training pipeline (image + text)

DeepSeek-OCR requires custom fields such as:

images

images_seq_mask

images_spatial_crop

This repo includes a complete DataCollator implementation that correctly builds these tensors.

✔ LoRA on language head only

Efficient LoRA config:

target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
r = 16
lora_alpha = 16


This reduces memory while improving text decoding quality.

✔ 5 万训练数据 + 2000 验证数据

Using priyank-m/chinese_text_recognition (HF public dataset), auto-converted into DeepSeek-OCR multi-modal format.

✔ Robust checkpointing for long training

Training runs for hours → this repo supports:

save_steps = 500

Resume from latest checkpoint

Safe loading of both base model + LoRA adapter

✔ Full evaluation suite

Includes:

1) Perplexity (HF evaluate)
2) CER (Character Error Rate)

A custom CER evaluator is provided to measure real OCR accuracy.

## 📂 3. Project Structure
deepseek-ocr-finetune/
│
├── data/
│   ├── ocr_dataset.py              # dataset loader & converter
│
├── collator/
│   ├── deepseek_ocr_collator.py    # critical multi-modal DataCollator
│
├── train/
│   ├── train_lora.py               # main training script with checkpointing
│
├── eval/
│   ├── eval_ppl.py                 # evaluation: perplexity
│   ├── eval_cer.py                 # evaluation: CER
│
├── README.md
└── requirements.txt

## 🧩 4. Data Format (DeepSeek-OCR Expected Structure)

Each sample is converted into:

{
  "messages": [
    {
      "role": "<|User|>",
      "content": "<image>"
    },
    {
      "role": "<|Assistant|>",
      "content": "识别后的文本内容"
    }
  ],
  "image": { "bytes": ... }
}

Why this matters

DeepSeek-OCR is multi-modal.
Only this structure correctly aligns:

Visual patches

Language tokens

Ignore mask

Assistant-only training regions

This repo includes a fully working converter.

## 🧠 5. Training Pipeline
Train with:
from transformers import TrainingArguments, Trainer
from unsloth import FastVisionModel

### Load model with custom remote code
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/DeepSeek-OCR",
    trust_remote_code = True,
    load_in_4bit = False
)

### Apply LoRA
model = FastVisionModel.get_peft_model(
    model,
    target_modules=[...],
    r=16,
    lora_alpha=16
)

TrainingArguments example:
training_args = TrainingArguments(
    output_dir = "./checkpoints",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    learning_rate = 1e-4,
    warmup_steps = 200,
    max_steps = 5000,

    save_strategy = "steps",
    save_steps = 500,
    save_total_limit = 3,

    eval_strategy = "steps",
    eval_steps = 500,
    logging_steps = 50,

    fp16 = True,
    remove_unused_columns = False,
)


Run training:

trainer = Trainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = DeepSeekOCRDataCollator(...),
    train_dataset = train_ds,
    eval_dataset = eval_ds,
    args = training_args,


trainer.train()

## 📈 6. Evaluation Results (Checkpoint: 1000 steps)

Evaluated on 50 / 100 / 200 / 1000 / 2000 samples.

Samples	CER
50	0.6957
100	0.7078
200	0.7006
1000	0.7253
2000	0.7006
Interpretation

Baseline CER ≈ 1.0（几乎不可用）

Fine-tuned CER ≈ 0.70

减少约 30% 字符级错误

模型的“补词/乱序”问题显著减少

长句结构保持更稳定

更适合作为结构化抽取的前置 OCR 模型

## 📉 7. Example Outputs

Example 1:

GT   : 党第一次代表大会
Pred : 第一次代表大会大党
CER  : 0.571


Example 2:

GT   : 海信LED46EC3
Pred : 海信LED46ec3
CER  : 0.09


Example 3:

GT   : ，也没有瘦高
Pred : 也没有瘦高的
CER  : 0.33


👉 明显减少冗余字、方向错误和补词。

## 🛠 8. Troubleshooting & Common Issues

This repo includes fixes for:

✔ transformers remote code loading failure
✔ DeepseekOCRConfig incompatible with AutoModel
✔ CPU/GPU mismatch in patch encoder
✔ dynamic_preprocess undefined
✔ DataCollator producing empty batches
✔ Accelerate mixed precision crashes
✔ HuggingFace removing unused columns (must disable)
✔ Vision token masks not aligned with labels

Every issue above has been solved and documented inside the repo.

## 🧪 9. Roadmap
✅ LoRA 微调（当前）
⬜ 支持全参数微调 (DeepSpeed ZeRO-2/3)
⬜ 支持模型在票据 OCR / 医疗 OCR 上继续扩展
⬜ Demo WebUI（Gradio）
⬜ ONNX / TensorRT 推理加速
⬜ Releasing real-world evaluation set

# 🔗 LoRA 权重获取

本仓库仅包含训练与评估代码，不直接托管大模型权重。

- DeepSeek-OCR 中文场景 LoRA 权重（step=1000）目前存放于个人云盘
