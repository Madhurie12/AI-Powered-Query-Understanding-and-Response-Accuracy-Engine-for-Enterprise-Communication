import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
import spacy
import os

# Load small English model for spacy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load CSV 
file_name = "messages_data.csv"  # Your file name
df = pd.read_csv(file_name)

# Use these two columns for evaluation
df.rename(columns={"question": "question", "response": "answer"}, inplace=True)

# Check if columns exist
if "question" not in df.columns or "answer" not in df.columns:
    print("Error: CSV must contain 'question' and 'response' columns.")
    exit(1)

df["question"] = df["question"].fillna("")
df["answer"] = df["answer"].fillna("")

# Load Embedding Model 
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def get_embedding(text):
    if not text.strip():
        return np.zeros(768)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

print("Generating embeddings...")
tqdm.pandas()
df["q_embed"] = df["question"].progress_apply(get_embedding)
df["a_embed"] = df["answer"].progress_apply(get_embedding)

# Cosine Similarity 
def get_cosine_sim(row):
    q_vec = row["q_embed"].reshape(1, -1)
    a_vec = row["a_embed"].reshape(1, -1)
    return cosine_similarity(q_vec, a_vec)[0][0]

df["answer_similarity_score"] = df.apply(get_cosine_sim, axis=1)

def label_relevance(score):
    if score >= 0.7:
        return "Relevant"
    elif score >= 0.4:
        return "Somewhat Relevant"
    else:
        return "Irrelevant"

df["answer_relevance"] = df["answer_similarity_score"].apply(label_relevance)

# Completeness Evaluation 
VAGUE_PHRASES = [
    "i don't know", "maybe", "i think", "i'm not sure", "hard to say",
    "it depends", "possibly", "can't say", "unsure", "idk"
]

def has_svo_structure(text):
    doc = nlp(text)
    has_subject = any(tok.dep_ in ("nsubj", "nsubjpass") for tok in doc)
    has_verb = any(tok.pos_ == "VERB" for tok in doc)
    has_object = any(tok.dep_ in ("dobj", "attr", "pobj") for tok in doc)
    return has_subject and has_verb and has_object

def is_vague_or_incomplete(text):
    text_lower = text.lower().strip()
    too_short = len(text.split()) < 5
    vague = any(phrase in text_lower for phrase in VAGUE_PHRASES)
    ends_with_connector = text_lower.split()[-1] in ["because", "so", "but", "although"]
    return too_short or vague or ends_with_connector

def evaluate_completeness(answer):
    if not answer.strip():
        return "Incomplete"
    if is_vague_or_incomplete(answer):
        return "Incomplete"
    if not has_svo_structure(answer):
        return "Incomplete"
    return "Complete"

print("Evaluating completeness...")
df["answer_completeness"] = df["answer"].progress_apply(evaluate_completeness)

# Success/Failure Labeling 
def evaluate_success(row):
    return row["answer_relevance"] == "Relevant" and row["answer_completeness"] == "Complete"

df["interaction_success"] = df.apply(evaluate_success, axis=1)

def fail_reason(row):
    if row["answer_relevance"] != "Relevant":
        return "irrelevant"
    elif row["answer_completeness"] != "Complete":
        return "incomplete"
    return "other"

df["fail_reason"] = df.apply(fail_reason, axis=1)

# Save Output 
output_file = "messages_with_success_failure.csv"
columns_to_save = [
    "question", "answer", "answer_similarity_score", "answer_relevance",
    "answer_completeness", "interaction_success", "fail_reason"
]
df.to_csv(output_file, columns=columns_to_save, index=False)
print(f"Results saved to: {output_file}")

# Final Report 
success_count = df["interaction_success"].sum()
fail_count = len(df) - success_count
total = len(df)
ratio = round(success_count / total, 2) if total else 0

print("\nSummary:")
print(f"✅ Successes: {success_count}")
print(f"❌ Failures: {fail_count}")
print(f"📊 Success Ratio: {ratio}")
