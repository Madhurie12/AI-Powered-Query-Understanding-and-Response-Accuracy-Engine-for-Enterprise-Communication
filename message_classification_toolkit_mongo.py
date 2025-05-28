
import pandas as pd
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict
from tqdm import tqdm
import random
from mongo_utils import get_collection

collection = get_collection("chatbot_db", "new_messages")
data = list(collection.find())
for doc in data:
    doc.pop("_id", None)

df = pd.DataFrame(data)
df["question"] = df["question"].fillna("")

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

vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["question"])
topic_model = KMeans(n_clusters=5, random_state=42).fit(tfidf_matrix)
df["topic_cluster"] = topic_model.labels_

nbq_lookup = defaultdict(list)
for _, row in df.iterrows():
    nbq_lookup[row["question_type_nlp"]].append(row["question"])

def suggest_nbq(q_class):
    pool = nbq_lookup.get(q_class, [])
    return random.choice(pool) if pool else ""

df["next_best_question"] = df["question_type_nlp"].apply(suggest_nbq)
topic_ranking = df["topic_cluster"].value_counts().to_dict()
df["topic_rank_score"] = df["topic_cluster"].map(topic_ranking)

output_path = "messages_classified_zero_shot_mongo.csv"
df.to_csv(output_path, index=False, encoding="utf-8")
print(f"Enhanced classified data saved to {output_path}")
