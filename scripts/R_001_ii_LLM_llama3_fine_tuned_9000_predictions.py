import pandas as pd
import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import classification_report


base_model = "NousResearch/Meta-Llama-3-8B-Instruct"
lora_model_path = "cc25k/models/Meta-Llama-3-8B-cc30k-9000"
test_file = "cc25k/test_244_gt.csv"

VALID_LABELS = {'positive', 'neutral', 'negative'}

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)


# Fix missing special tokens
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = '[PAD]'

if tokenizer.eos_token is None:
    tokenizer.eos_token = tokenizer.pad_token


# Load base + LoRA model
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
base = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
model = PeftModel.from_pretrained(base, lora_model_path)
model.eval()

# Load test data
df_test = pd.read_csv(test_file)


def build_prompt(context):
    system_msg = (
        "You are a helpful assistant that classifies scientific citation contexts based on reproducibility-oriented sentiment.\n\n"
        "Definitions:\n"
        "- **Positive**: Suggests successful reproducibility or replicability — reuse of data/code, replication of experiments, or references to tools/software used again.\n"
        "- **Negative**: Suggests irreproducibility or irreplicability — missing code/data, failed reproduction attempts, or criticism of reproducibility.\n"
        "- **Neutral**: Merely cites the paper without any comment on reproducibility or replication.\n\n"
        "Only respond with one of the following labels exactly: **Positive**, **Negative**, **Neutral**."
    )

    return (
        f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n"
        f"Context:\n{context}\n"
        f"What is the reproducibility-oriented sentiment of this citation context?\n"
        f"[/INST]\nAnswer:"
    )



# def build_prompt(context):
#     system_msg = (
#         "You are a helpful assistant that classifies scientific citation contexts based on reproducibility-oriented sentiment.\n\n"
#         "Definitions:\n"
#         "- **Positive**: Suggests successful reproducibility or replicability — reuse of data/code, replication of experiments, or references to tools/software used again.\n"
#         "- **Negative**: Suggests irreproducibility or irreplicability — missing code/data, failed reproduction attempts, or criticism of reproducibility.\n"
#         "- **Neutral**: Merely cites the paper without any comment on reproducibility or replication.\n\n"
#         "Only respond with one of the following labels exactly: **Positive**, **Negative**, **Neutral**."
#     )

#     examples = """
# Examples:
# Context: Our work is built upon the official setup of EFDMix (Zhang et al., 2022).
# Answer: Positive

# Context: For LC estimation, we use Monte Carlo (MC) approach (Yan & Procaccia, 2020) as the baseline.
# Answer: Positive

# Context: It was therefore unclear how to get the dataset labels for the tSNE latent space visualization in Figure 4 as it is not mentioned in the paper [1].
# Answer: Negative

# Context: We also tried MBPO [35], but we found that this method takes too much memory and could not finish any test.
# Answer: Negative

# Context: In vision-and-language tasks, there has been some recent advancements, especially for image captioning [21, 23, 47, 59].
# Answer: Neutral

# Context: Contrarily, MAE [21] reconstructs raw pixels of the image explicitly.
# Answer: Neutral
# """

#     return (
#         f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n"
#         f"{examples}\n\n"
#         f"Context: {context}\n"
#         f"What is the reproducibility-oriented sentiment of this citation context?\n"
#         f"[/INST]\nAnswer:"
#     )


# to support the few shot prompting
def extract_clean_label(response_text):
    matches = re.findall(r'Answer:\s*(Positive|Neutral|Negative)', response_text, re.IGNORECASE)
    if matches:
        return matches[-1].capitalize()  # Get the last matched label
    return "---"


# EOS token IDs
eos_tokens = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

# Generate predictions
predictions = []
raw_predictions = []
for context in df_test["input_context"]:

    inputs = tokenizer(build_prompt(context), return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=10, #5
        temperature=0.8,
        do_sample=True,
        top_k=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    response = extract_clean_label(generated)
    
    predictions.append(response)
    raw_predictions.append(generated)

# Evaluation
df_test["predicted"] = predictions
df_test["raw_predictions"] = raw_predictions
report = classification_report(df_test["label_gt"].str.lower(), df_test["predicted"].str.lower(), digits=3)
print(report)
print("LLAMA 9000 - zero shot - manaully verfied negatvies included")
df_test.to_csv(r"cc25k/cc30k_244_predictions_llama3_fine_tuned_9000.csv", index=False)
