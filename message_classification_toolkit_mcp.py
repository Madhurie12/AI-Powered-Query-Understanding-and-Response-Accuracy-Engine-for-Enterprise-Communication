import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict
from tqdm import tqdm
import random

# Load CSV
csv_path = r"C:\Users\admin\OneDrive\Desktop\messages_data.csv"
try:
    df = pd.read_csv(csv_path, encoding="utf-8")
except FileNotFoundError:
    print(f"Error: {csv_path} not found.")
    exit(1)

# Check for valid columns
if "question" not in df.columns or df["question"].isnull().all():
    raise ValueError("CSV must contain a non-empty 'question' column.")

df["question"] = df["question"].fillna("")

# Sentiment Analysis 
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["question"].apply(get_sentiment)

# User Profile Segmentation 
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

df["user_profile"] = df["question"].apply(segment_user_profile)

# Zero-Shot Classification (MCP) 
print("Loading zero-shot classifier...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)
labels = ["Informational", "Transactional", "Conversational"]

def zero_shot_classify(text):
    if not isinstance(text, str) or not text.strip():
        return "Unknown"
    result = classifier(text, candidate_labels=labels)
    return result["labels"][0]

print("Classifying question types using zero-shot MCP...")
tqdm.pandas()
df["question_type_nlp"] = df["question"].progress_apply(zero_shot_classify)

# TF-IDF Topic Clustering 
vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["question"])
topic_model = KMeans(n_clusters=5, random_state=42).fit(tfidf_matrix)
df["topic_cluster"] = topic_model.labels_

# Next Best Question Suggestion 
nbq_lookup = defaultdict(list)
for _, row in df.iterrows():
    nbq_lookup[row["question_type_nlp"]].append(row["question"])

def suggest_nbq(q_class):
    pool = nbq_lookup.get(q_class, [])
    return random.choice(pool) if pool else ""

df["next_best_question"] = df["question_type_nlp"].apply(suggest_nbq)

# Topic Ranking 
topic_ranking = df["topic_cluster"].value_counts().to_dict()
df["topic_rank_score"] = df["topic_cluster"].map(topic_ranking)

# Sample Output 
print("\nSample Classifications (5 per class):")
for cls in labels:
    samples = df[df["question_type_nlp"] == cls][["question", "question_type_nlp"]].head(5)
    print(f"\n{cls}:\n{samples}")

print("\nQuestion Type Distribution:")
print(df["question_type_nlp"].value_counts())

#  Save Output 
output_path = r"C:\Users\admin\OneDrive\Desktop\messages_classified_zero_shot.csv"
df.to_csv(output_path, index=False, encoding="utf-8")
print(f"\nEnhanced classified data saved to {output_path}")
