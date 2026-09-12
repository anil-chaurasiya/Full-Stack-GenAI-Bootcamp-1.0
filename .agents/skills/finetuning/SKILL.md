---
name: finetuning
description: Fine-tune LLMs using Hugging Face Transformers, Unsloth, LoRA, QLoRA, RLHF, and DPO techniques. Covers instruction tuning, preference alignment, and multimodal fine-tuning.
---

# Fine-Tuning Skill

Use this skill when the user wants to fine-tune, adapt, or train language models.

## Fine-Tuning Methods

### 1. Standard Fine-Tuning (Hugging Face)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

dataset = load_dataset("your_dataset")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    save_strategy="epoch",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

trainer.train()
```

### 2. LoRA (Low-Rank Adaptation)
```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # Rank
    lora_alpha=32,           # Scaling factor
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # Model-specific
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### 3. QLoRA (Quantized LoRA)
```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
```

### 4. Fine-Tuning with Unsloth (Faster)
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

### 5. DPO (Direct Preference Optimization)
```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    output_dir="./dpo_results",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    beta=0.1,  # KL penalty coefficient
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

## Dataset Formats

### Instruction Tuning Format
```json
{
  "instruction": "Summarize the following text",
  "input": "Long text here...",
  "output": "Summary here..."
}
```

### Chat Format (Preferred)
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Question here"},
    {"role": "assistant", "content": "Answer here"}
  ]
}
```

### Preference Data (for DPO)
```json
{
  "prompt": "User question",
  "chosen": "Preferred response",
  "rejected": "Non-preferred response"
}
```

## Hardware Requirements

| Method       | VRAM Required  | Recommended GPU        |
|--------------|----------------|------------------------|
| Full FT (7B) | 28+ GB         | A100 / 4x A10G        |
| LoRA (7B)    | 16+ GB         | T4 / A10G              |
| QLoRA (7B)   | 6–8 GB         | T4 (free Colab)        |
| Unsloth      | 5–6 GB         | T4 (free Colab)        |

## Tips
- Always use **QLoRA or Unsloth** for resource-constrained environments
- Use **Google Colab** (free T4 GPU) if no local GPU available
- Save adapters separately: `model.save_pretrained("./adapter")`
- Merge adapters for deployment: `model.merge_and_unload()`
- Monitor training with `wandb` or TensorBoard
