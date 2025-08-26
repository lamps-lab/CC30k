import os
import re
import numpy as np
import pandas as pd
import faiss #faiss-cpu
from sklearn.metrics import classification_report
from openai import OpenAI
from sentence_transformers import SentenceTransformer 
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()

# ========= Config =========
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
train_file = "cc25k/new_extended_train_set_llm_9000.csv"       
test_file = "cc25k/test_244_gt.csv"  
output_file = "cc25k/cc30k_244_predictions_gpt4o_RAG_9000.csv"

VALID_LABELS = {"Positive", "Neutral", "Negative"}
K = 5  # top-k examples for RAG

# ========= Load Training & Test Data =========
df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

# ========= Create Embeddings for Training Data =========
print("Generating embeddings for training data...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
train_embeddings = embed_model.encode(df_train["text"].tolist(), convert_to_tensor=False)

# ========= Build FAISS Index =========
dimension = train_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(train_embeddings))

# ========= Retrieve Top-K Similar Examples =========
def retrieve_examples(query, k=K):
    query_embedding = embed_model.encode([query], convert_to_tensor=False)
    D, I = index.search(np.array(query_embedding), k)
    examples = []
    for idx in I[0]:
        examples.append((df_train["text"].iloc[idx], df_train["label"].iloc[idx]))
    return examples

# ========= Prompt Builder with RAG =========
def build_rag_prompt(context):
    system_msg = (
        "You are a helpful assistant that classifies scientific citation contexts based on reproducibility-oriented sentiment. Use the following definitions to guide your classification:\n\n"
        "- Positive: The context suggests successful reproducibility or replicability, such as the reuse of data, code, or concepts from the cited paper. It may include terms such as reproduce, replicate, or repeat the experiments, or references to the software or processes from the cited paper being used for pre-processing or comparison.\n"
        "- Negative: The context hints at irreproducibility or irreplicability, such as the unavailability of the cited paper’s data or code, or unsuccessful attempts to reproduce the results.\n"
        "- Neutral: The context simply mentions (cites) the cited paper without providing any hints about reproducibility. These contexts lack any indication of attempts to run the implementation or verify the results.\n\n"
        "Respond only with one of the following labels: Positive, Negative, or Neutral.\n\n"
        "Here are some labeled examples:\n"
    )
    examples = retrieve_examples(context, k=K)
    for i, (ex_text, ex_label) in enumerate(examples, start=1):
        system_msg += f"Example {i}:\nCitation Context: {ex_text}\nLabel: {ex_label}\n\n"
    system_msg += f"Now classify this new citation context:\n{context}\nAnswer:"
    return system_msg

# ========= Label Extractor =========
def extract_clean_label(response_text):
    match = re.search(r"(Positive|Neutral|Negative)", response_text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return "---"

# ========= Inference =========
predictions = []
raw_predictions = []

print("Running GPT-4o + RAG inference...")
for context in df_test["input_context"]:
    prompt = build_rag_prompt(context)
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

# ========= Evaluation =========
df_test["predicted"] = predictions
df_test["raw_predictions"] = raw_predictions
report = classification_report(
    df_test["label_gt"].str.lower(),
    df_test["predicted"].str.lower(),
    digits=3
)
print(report)

# ========= Save Predictions =========
df_test.to_csv(output_file, index=False)
print(f"gpt4o_RAG_9000 (new 9000 WITH manually verified negatives) Predictions saved to {output_file}")


# import os
# import re
# import numpy as np
# import pandas as pd
# import faiss  # faiss-cpu
# from sklearn.metrics import classification_report
# from openai import OpenAI
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv
# from collections import defaultdict

# # Load variables from .env into environment
# load_dotenv()

# # ========= Config =========
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# train_file = "cc25k/new_extended_train_set_llm_9000.csv"       
# test_file = "cc25k/test_244_gt.csv"    
# output_file = "cc25k/cc30k_244_predictions_gpt4o_RAG_fewshot_9000.csv"

# VALID_LABELS = {"Positive", "Neutral", "Negative"}
# K = 5  # top-k retrieved examples for RAG
# FEW_SHOT_PER_CLASS = 3  # how many static few-shot examples per class

# # ========= Load Training & Test Data =========
# df_train = pd.read_csv(train_file)
# df_test = pd.read_csv(test_file)

# # ========= Prepare few-shot examples (balanced 3 per class) =========
# few_shot_examples = []
# for label in VALID_LABELS:
#     subset = df_train[df_train["label"].str.lower() == label.lower()]
#     sampled = subset.sample(n=FEW_SHOT_PER_CLASS, random_state=42)
#     for _, row in sampled.iterrows():
#         few_shot_examples.append((row["text"], row["label"]))

# # ========= Create Embeddings for Training Data =========
# print("Generating embeddings for training data...")
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")
# train_texts = df_train["text"].tolist()
# train_labels = df_train["label"].tolist()
# train_embeddings = embed_model.encode(train_texts, convert_to_tensor=False)

# # ========= Build FAISS Index =========
# dimension = train_embeddings[0].shape[0]
# index = faiss.IndexFlatL2(dimension)
# index.add(np.array(train_embeddings).astype("float32"))

# # ========= Retrieve Top-K Similar Examples =========
# def retrieve_examples(query, k=K):
#     query_embedding = embed_model.encode([query], convert_to_tensor=False)
#     D, I = index.search(np.array(query_embedding).astype("float32"), k)
#     examples = []
#     for idx in I[0]:
#         examples.append((train_texts[idx], train_labels[idx]))
#     return examples

# # ========= Prompt Builder with Few-shot + RAG =========
# def build_rag_fewshot_prompt(context):
#     base = (
#         "You are a helpful assistant that classifies scientific citation contexts based on reproducibility-oriented sentiment. "
#         "Use the following definitions to guide your classification:\n\n"
#         "- Positive: The context suggests successful reproducibility or replicability, such as the reuse of data, code, or concepts from the cited paper. "
#         "It may include terms such as reproduce, replicate, or repeat the experiments, or references to the software or processes from the cited paper being used for pre-processing or comparison.\n"
#         "- Negative: The context hints at irreproducibility or irreplicability, such as the unavailability of the cited paper’s data or code, or unsuccessful attempts to reproduce the results.\n"
#         "- Neutral: The context simply mentions (cites) the cited paper without providing any hints about reproducibility. These contexts lack any indication of attempts to run the implementation or verify the results.\n\n"
#         "Respond only with one of the following labels: Positive, Negative, or Neutral.\n\n"
#     )

#     # Few-shot examples (balanced)
#     base += "Few-shot examples:\n"
#     for i, (ex_text, ex_label) in enumerate(few_shot_examples, start=1):
#         base += f"Example {i}:\nCitation Context: {ex_text}\nLabel: {ex_label}\n\n"

#     # Retrieved examples
#     retrieved = retrieve_examples(context, k=K)
#     base += "Retrieved similar examples:\n"
#     for i, (ex_text, ex_label) in enumerate(retrieved, start=1):
#         base += f"Retrieved {i}:\nCitation Context: {ex_text}\nLabel: {ex_label}\n\n"

#     # Query
#     base += f"Now classify this new citation context:\n{context}\nAnswer:"
#     return base

# # ========= Label Extractor =========
# def extract_clean_label(response_text):
#     match = re.search(r"(Positive|Neutral|Negative)", response_text, re.IGNORECASE)
#     if match:
#         return match.group(1).capitalize()
#     return "---"

# # ========= Inference =========
# predictions = []
# raw_predictions = []

# print("Running GPT-4o + RAG + few-shot inference...")
# for context in df_test["input_context"]:
#     prompt = build_rag_fewshot_prompt(context)
#     try:
#         completion = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.2, 
#             max_tokens=10
#         )
#         generated = completion.choices[0].message.content.strip()
#     except Exception as e:
#         generated = f"ERROR: {e}"
    
#     label = extract_clean_label(generated)
#     predictions.append(label)
#     raw_predictions.append(generated)

# # ========= Evaluation =========
# df_test["predicted"] = predictions
# df_test["raw_predictions"] = raw_predictions
# report = classification_report(
#     df_test["label_gt"].str.lower(),
#     df_test["predicted"].str.lower(),
#     digits=3
# )
# print(report)

# # ========= Save Predictions =========
# df_test.to_csv(output_file, index=False)
# print(f"gpt4o_RAG_fewshot_9000 (new 9000 WITH manually verified negatives) Predictions saved to {output_file}")

