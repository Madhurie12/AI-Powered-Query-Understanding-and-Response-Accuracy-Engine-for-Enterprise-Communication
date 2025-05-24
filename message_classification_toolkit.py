import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from collections import defaultdict
from tqdm import tqdm
import os
import random

# Load Data
csv_path = r"C:\Users\admin\OneDrive\Desktop\messages_data.csv"
try:
    messages_df = pd.read_csv(csv_path, encoding="utf-8")
except FileNotFoundError:
    print(f"Error: {csv_path} not found. Run setup_data.py first.")
    exit(1)

# Filter questions
messages_df["question"] = messages_df["question"].fillna("")
if "role" in messages_df.columns:
    questions_df = messages_df[messages_df["role"] == "User"].copy()
else:
    questions_df = messages_df.copy()

print(f"Total questions loaded: {len(questions_df)}")
if len(questions_df) < 5000:
    print("Warning: Expected ~5,367 questions. Check messages_data.csv.")

# ASSIGNMENT POINT 1: Sentiment Analysis 
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

questions_df["sentiment"] = questions_df["question"].apply(get_sentiment)

# ASSIGNMENT POINT 2: User Intent/Profile Segmentation 
def segment_user_profile(question):
    q = question.lower()
    if any(x in q for x in ["how", "why", "what", "explain"]):
        return "Curious"
    elif any(x in q for x in ["book", "schedule", "order", "can you"]):
        return "Action-Oriented"
    elif any(x in q for x in ["hi", "hey", "hello", "yo", "morning"]):
        return "Casual"
    else:
        return "General"

questions_df["user_profile"] = questions_df["question"].apply(segment_user_profile)

# Load DistilBERT 
print("Loading DistilBERT...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
device = torch.device("cpu")
model.to(device)
model.eval()

# Generate embeddings
def get_embedding(text):
    if not text:
        return np.zeros(768)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

print("Generating embeddings...")
tqdm.pandas()
questions_df["embedding"] = questions_df["question"].progress_apply(get_embedding)
embeddings = np.vstack(questions_df["embedding"].values)

# Clustering 
print("Clustering questions...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
questions_df["cluster"] = kmeans.fit_predict(embeddings)

# Manual mapping after inspection (can adjust based on output)
cluster_to_class = {
    0: "Informational",
    1: "Conversational",
    2: "Transactional"
}

def refine_class(question, cluster):
    question = question.lower()
    if cluster == 0 and any(kw in question for kw in ["book", "order", "reserve"]):
        return "Transactional"
    elif cluster == 1 and any(kw in question for kw in ["tell", "what", "how"]):
        return "Informational"
    return cluster_to_class[cluster]

questions_df["question_type_nlp"] = questions_df.apply(
    lambda x: refine_class(x["question"], x["cluster"]), axis=1
)

# ASSIGNMENT POINT 3: Topic Modeling (TF-IDF + KMeans) 
vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
tfidf_matrix = vectorizer.fit_transform(questions_df["question"])
topic_model = KMeans(n_clusters=5, random_state=42).fit(tfidf_matrix)
questions_df["topic_cluster"] = topic_model.labels_

# ASSIGNMENT POINT 4: Next Best Question Suggestion 
nbq_lookup = defaultdict(list)
for _, row in questions_df.iterrows():
    nbq_lookup[row["question_type_nlp"]].append(row["question"])

def suggest_nbq(q_class):
    pool = nbq_lookup.get(q_class, [])
    return random.choice(pool) if pool else ""

questions_df["next_best_question"] = questions_df["question_type_nlp"].apply(suggest_nbq)

# ASSIGNMENT POINT 5: Knowledge Ranking Stub 
topic_ranking = questions_df["topic_cluster"].value_counts().to_dict()
questions_df["topic_rank_score"] = questions_df["topic_cluster"].map(topic_ranking)

# Sample Output for Validation 
print("\nSample Classifications (10 per class):")
for cls in ["Informational", "Transactional", "Conversational"]:
    samples = questions_df[questions_df["question_type_nlp"] == cls][["question", "question_type_nlp"]].head(10)
    print(f"{cls}:\n{samples}\n")

print("\nQuestion Type Distribution:")
print(questions_df["question_type_nlp"].value_counts())

# Save Final Output 
output_path = r"C:\Users\admin\OneDrive\Desktop\messages_classified_nlp.csv"
questions_df.to_csv(output_path, index=False, encoding="utf-8")
print(f"Enhanced classified data saved to {output_path}")
