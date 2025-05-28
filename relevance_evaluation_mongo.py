
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
from mongo_utils import get_collection

collection = get_collection("chatbot_db", "new_messages")
data = list(collection.find())
for doc in data:
    doc.pop("_id", None)

df = pd.DataFrame(data)
df["question"] = df["question"].fillna("")
df["answer"] = df["answer"].fillna("")

print("Loaded columns:", df.columns.tolist())

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
device = torch.device("cpu")
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

print("Generating question & answer embeddings...")
tqdm.pandas()
df["q_embed"] = df["question"].progress_apply(get_embedding)
df["a_embed"] = df["answer"].progress_apply(get_embedding)

def get_cosine_sim(row):
    q_vec = row["q_embed"].reshape(1, -1)
    a_vec = row["a_embed"].reshape(1, -1)
    return cosine_similarity(q_vec, a_vec)[0][0]

print("Computing similarity scores...")
df["answer_similarity_score"] = df.apply(get_cosine_sim, axis=1)

def label_relevance(score):
    if score >= 0.7:
        return "Relevant"
    elif score >= 0.4:
        return "Somewhat Relevant"
    else:
        return "Irrelevant"

df["answer_relevance"] = df["answer_similarity_score"].apply(label_relevance)

output_path = "messages_with_relevance_mongo.csv"
df.drop(columns=["q_embed", "a_embed"]).to_csv(output_path, index=False, encoding="utf-8")
print(f"Relevance evaluation saved to: {output_path}")
