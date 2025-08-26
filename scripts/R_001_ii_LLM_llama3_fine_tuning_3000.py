import os
import logging
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

# ========== Model and Tokenizer ==========
model_name = "NousResearch/Meta-Llama-3-8B-Instruct"
output_dir = "cc25k/models/Meta-Llama-3-8B-cc30k-3000"


# ========== Logging Setup ==========
log_dir = f"{output_dir}/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting fine-tuning job")


# ========== Dataset ==========
data_path = r"cc25k/llm_fine_tune_dataset_llama3_3000.jsonl"
logging.info(f"Loading dataset from {data_path}")
dataset = load_dataset("json", data_files=data_path, split="train")
dataset = dataset.train_test_split(test_size=0.1)


logging.info(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")

# ========== LoRA Config ==========
lora_config = LoraConfig(
    r=16, #8 , 16, 32
    lora_alpha=32, #16, 32, 64
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# ========== Training Arguments ==========
os.makedirs(output_dir, exist_ok=True)

# ========== SFT Trainer Config ==========
sft_config = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=15, #25 
    learning_rate=2e-4, # 2e-5
    logging_dir="./logs/tensorboard",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    max_seq_length=1024,
    dataset_text_field="prompt",
    fp16=True,
    report_to="none",
)

# ========== Trainer ==========
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=lora_config,
    args=sft_config,
)

# ========== Train ==========
try:
    logging.info("Training started")
    trainer.train()
    logging.info("Training completed successfully")
except Exception as e:
    logging.exception(f"Training failed with exception: {e}")

# ========== Save Model ==========
trainer.save_model(output_dir)
logging.info(f"Model saved to {output_dir}")

