import pandas as pd
import re
import os
from sklearn.metrics import classification_report
from openai import OpenAI
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()

# ========== Config ==========
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
train_file = "cc25k/new_extended_train_set_llm_3000.csv"   
test_file = "cc25k/test_244_gt.csv"
output_file = "cc25k/cc30k_244_predictions_gpt4o_original.csv"

VALID_LABELS = {'Positive', 'Neutral', 'Negative'}

FEW_SHOT_PER_CLASS = 3  

# ========== Load Test Data ==========
df_test = pd.read_csv(test_file)

df_train = pd.read_csv(train_file)

# ========= Prepare few-shot examples (balanced 3 per class) =========
few_shot_examples = []
for label in VALID_LABELS:
    subset = df_train[df_train["label"].str.lower() == label.lower()]
    sampled = subset.sample(n=FEW_SHOT_PER_CLASS, random_state=42)
    for _, row in sampled.iterrows():
        few_shot_examples.append((row["text"], row["label"]))


# ========= Prompt Builder with Few-shot + RAG =========
def build_prompt(context):
    base = (
        "You are a helpful assistant that classifies scientific citation contexts based on reproducibility-oriented sentiment. "
        "Use the following definitions to guide your classification:\n\n"
        "- Positive: The context suggests successful reproducibility or replicability, such as the reuse of data, code, or concepts from the cited paper. "
        "It may include terms such as reproduce, replicate, or repeat the experiments, or references to the software or processes from the cited paper being used for pre-processing or comparison.\n"
        "- Negative: The context hints at irreproducibility or irreplicability, such as the unavailability of the cited paper’s data or code, or unsuccessful attempts to reproduce the results.\n"
        "- Neutral: The context simply mentions (cites) the cited paper without providing any hints about reproducibility. These contexts lack any indication of attempts to run the implementation or verify the results.\n\n"
        "Respond only with one of the following labels: Positive, Negative, or Neutral.\n\n"
    )

    # Few-shot examples (balanced)
    base += "Few-shot examples:\n"
    for i, (ex_text, ex_label) in enumerate(few_shot_examples, start=1):
        base += f"Example {i}:\nCitation Context: {ex_text}\nLabel: {ex_label}\n\n"

    # Query
    base += f"Now classify this new citation context:\n{context}\nAnswer:"
    return base

# ========== Label Extractor ==========
def extract_clean_label(response_text):
    match = re.search(r'(Positive|Neutral|Negative)', response_text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return "---"


# ========== Inference ==========
predictions = []
raw_predictions = []

for context in df_test["input_context"]:
    prompt = build_prompt(context)
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10
        )
        generated = completion.choices[0].message.content.strip()
    except Exception as e:
        generated = f"ERROR: {e}"
    
    label = extract_clean_label(generated)
    predictions.append(label)
    raw_predictions.append(generated)

# ========== Evaluation ==========
df_test["predicted"] = predictions
df_test["raw_predictions"] = raw_predictions
report = classification_report(df_test["label_gt"].str.lower(), df_test["predicted"].str.lower(), digits=3)
print(report)
print("GPT-4o original (base) with few shot")

df_test.to_csv(output_file, index=False)


# import pandas as pd
# import torch
# import re
# import os
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from peft import PeftModel
# from sklearn.metrics import classification_report

# # ========== Paths ==========
# base_model = "NousResearch/Meta-Llama-3-8B-Instruct"
# test_file = "cc25k/test_244_gt.csv"
# output_file = "cc25k/cc30k_244_predictions_gpt4o_original.csv"

# VALID_LABELS = {'positive', 'neutral', 'negative'}

# # ========== Load Tokenizer ==========
# tokenizer = AutoTokenizer.from_pretrained(base_model)

# # Fix missing special tokens
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#     tokenizer.pad_token = '[PAD]'

# if tokenizer.eos_token is None:
#     tokenizer.eos_token = tokenizer.pad_token

# # ========== Load Base Model Only ==========
# bnb_config = BitsAndBytesConfig(load_in_4bit=True)
# model = AutoModelForCausalLM.from_pretrained(
#     base_model,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     quantization_config=bnb_config
# )
# model.eval()

# # ========== Load Test Data ==========
# df_test = pd.read_csv(test_file)

# # ========== Prompt Template ==========
# def build_prompt(context):
#     return f"""You are a helpful assistant that classifies scientific citation contexts based on reproducibility-oriented sentiment. Use the following definitions to guide your classification:

# - Positive: The context suggests successful reproducibility or replicability, such as the reuse of data, code, or concepts from the cited paper. It may include terms such as reproduce, replicate, or repeat the experiments, or references to the software or processes from the cited paper being used for pre-processing or comparison.

# - Negative: The context hints at irreproducibility or irreplicability, such as the unavailability of the cited paper’s data or code, or unsuccessful attempts to reproduce the results.

# - Neutral: The context simply mentions (cites) the cited paper without providing any hints about reproducibility. These contexts lack any indication of attempts to run the implementation or verify the results.

# Respond only with one of the following labels: Positive, Neutral, or Negative.

# Context:
# {context}

# Answer:"""


# # ========== Label Extractor ==========
# def extract_clean_label(response_text):
#     match = re.search(r'Answer:\s*(Positive|Neutral|Negative)', response_text, re.IGNORECASE)
#     if match:
#         return match.group(1).capitalize()
#     return "---"

# # ========== Inference ==========
# predictions = []
# raw_predictions = []

# for context in df_test["input_context"]:
#     prompt = build_prompt(context)
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=20,
#         do_sample=False,
#         pad_token_id=tokenizer.pad_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#     )
    
#     generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     label = extract_clean_label(generated)
    
#     predictions.append(label)
#     raw_predictions.append(generated)

# # ========== Evaluation ==========
# df_test["predicted"] = predictions
# df_test["raw_predictions"] = raw_predictions
# report = classification_report(df_test["label_gt"].str.lower(), df_test["predicted"].str.lower(), digits=3)
# print(report)

# df_test.to_csv(output_file, index=False)

