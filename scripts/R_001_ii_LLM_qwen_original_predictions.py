import pandas as pd
import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import classification_report
from dotenv import load_dotenv


# ========== Config ==========
base_model = "Qwen/Qwen1.5-7B-Chat"
test_file = "cc25k/test_244_gt.csv"

# ========== Authentication ==========
load_dotenv()
auth_token = os.getenv("HF_TOKEN")

# ========== Tokenizer ==========
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, token=auth_token)

# Fix pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ========== Model ==========
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization_config=bnb_config,
    token=auth_token
)
model.eval()

# ========== Data ==========
df_test = pd.read_csv(test_file)

# # ========== Prompt Template ==========
def build_prompt(context):
    return f"""<|im_start|>system
You are a helpful assistant that classifies scientific citation contexts based on reproducibility-oriented sentiment. Use the following definitions to guide your classification:

- Positive: The context suggests successful reproducibility or replicability, such as the reuse of data, code, or concepts from the cited paper. It may include terms such as reproduce, replicate, or repeat the experiments, or references to the software or processes from the cited paper being used for pre-processing or comparison.

- Negative: The context hints at irreproducibility or irreplicability, such as the unavailability of the cited paper’s data or code, or unsuccessful attempts to reproduce the results.

- Neutral: The context simply mentions (cites) the cited paper without providing any hints about reproducibility. These contexts lack any indication of attempts to run the implementation or verify the results.

Respond only with one of the following labels: Positive, Neutral, or Negative.<|im_end|>
<|im_start|>user
{context}
<|im_end|>
<|im_start|>assistant
Answer:"""


# def build_prompt(context):
#     return f"""<|im_start|>system
# You are a helpful assistant that classifies scientific citation contexts based on reproducibility-oriented sentiment.

# Definitions:
# - **Positive**: The context indicates successful reproducibility or replicability — reuse of data/code, replication of experiments, or tools/software from the cited paper used again.
# - **Negative**: The context suggests irreproducibility or irreplicability — missing data/code, failed reproduction attempts, or critique of reproducibility.
# - **Neutral**: The citation simply mentions the paper without commenting on reproducibility or replication.

# Respond with exactly one of the following labels: **Positive**, **Negative**, or **Neutral**.
# <|im_end|>
# <|im_start|>user
# citation context: Our work is built upon the official setup of EFDMix (Zhang et al., 2022).
# What is the reproducibility-oriented sentiment of this citation context?
# <|im_end|>
# <|im_start|>assistant
# Answer: Positive
# <|im_end|>
# <|im_start|>user
# citation context: In vision-and-language tasks, there has been some recent advancements, especially for image captioning [21, 23, 47, 59].
# What is the reproducibility-oriented sentiment of this citation context?
# <|im_end|>
# <|im_start|>assistant
# Answer: Neutral
# <|im_end|>
# <|im_start|>user
# citation context: We also tried MBPO [35], but we found that this method takes too much memory and could not finish any test.
# What is the reproducibility-oriented sentiment of this citation context?
# <|im_end|>
# <|im_start|>assistant
# Answer: Negative
# <|im_end|>
# <|im_start|>user
# citation context: {context}
# What is the reproducibility-oriented sentiment of this citation context?
# <|im_end|>
# <|im_start|>assistant
# Answer:"""

# ========== Label Extractor ==========
def extract_clean_label(response_text):
    matches = re.findall(r'Answer:\s*(Positive|Neutral|Negative)', response_text, re.IGNORECASE)
    if matches:
        return matches[-1].capitalize()  # Return the last match (actual inference)
    return "---"

# ========== Inference ==========
predictions = []
raw_predictions = []

for context in df_test["input_context"]:
    prompt = build_prompt(context)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    label = extract_clean_label(decoded)
    predictions.append(label)
    raw_predictions.append(decoded)

# ========== Evaluation ==========
df_test["predicted"] = predictions
df_test["raw_predictions"] = raw_predictions
report = classification_report(df_test["label_gt"].str.lower(), df_test["predicted"].str.lower(), digits=3)
print(report)
print("qwen original with few shot")

df_test.to_csv(r"cc25k/cc30k_244_predictions_qwen_original.csv", index=False)
