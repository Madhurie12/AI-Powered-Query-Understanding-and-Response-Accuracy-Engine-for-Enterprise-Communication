import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch

# Load CSV with 'question' and 'answer'-type columns 
csv_path = r"C:\Users\admin\OneDrive\Desktop\messages_data.csv"
try:
    df = pd.read_csv(csv_path, encoding="utf-8")
except FileNotFoundError:
    print(f"Error: {csv_path} not found.")
    exit(1)

# Inspect columns to detect answer column 
print("Loaded columns:", df.columns.tolist())
answer_col = None
for col in df.columns:
    if col.lower() in ["answer", "response", "bot_reply", "chatbot_answer"]:
        answer_col = col
        break

if not answer_col:
    raise ValueError("No answer column found. Expected one of: 'answer', 'response', 'bot_reply', 'chatbot_answer'")

# Standardize column names 
df.rename(columns={answer_col: "answer"}, inplace=True)
if "question" not in df.columns:
    raise ValueError("'question' column not found in the data.")

# Clean Nulls 
df["question"] = df["question"].fillna("")
df["answer"] = df["answer"].fillna("")

# Load DistilBERT 
print("Loading DistilBERT for embedding...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
device = torch.device("cpu")
model.to(device)
model.eval()

# Embedding Function 
def get_embedding(text):
    if not text.strip():
        return np.zeros(768)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

# Generate Embeddings 
print("Generating question & answer embeddings...")
tqdm.pandas()
df["q_embed"] = df["question"].progress_apply(get_embedding)
df["a_embed"] = df["answer"].progress_apply(get_embedding)

# Cosine Similarity 
def get_cosine_sim(row):
    q_vec = row["q_embed"].reshape(1, -1)
    a_vec = row["a_embed"].reshape(1, -1)
    return cosine_similarity(q_vec, a_vec)[0][0]

print("Computing similarity scores...")
df["answer_similarity_score"] = df.apply(get_cosine_sim, axis=1)

# Label Relevance 
def label_relevance(score):
    if score >= 0.7:
        return "Relevant"
    elif score >= 0.4:
        return "Somewhat Relevant"
    else:
        return "Irrelevant"

df["answer_relevance"] = df["answer_similarity_score"].apply(label_relevance)

# Sample Output 
print("\nSample Results:")
for idx, row in df.head(10).iterrows():
    print(f"Question: {row['question']}\nAnswer: {row['answer']}")
    print(f"Relevance Score: {row['answer_similarity_score']:.4f}")
    print(f"Relevance Label: {row['answer_relevance']}\n")

# Save to CSV 
output_path = r"C:\Users\admin\OneDrive\Desktop\messages_with_relevance.csv"
df.drop(columns=["q_embed", "a_embed"]).to_csv(output_path, index=False, encoding="utf-8")
print(f"\nRelevance evaluation saved to: {output_path}")
