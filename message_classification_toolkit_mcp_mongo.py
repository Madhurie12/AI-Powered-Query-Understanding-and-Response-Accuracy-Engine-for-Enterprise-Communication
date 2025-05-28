
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from collections import defaultdict
from tqdm import tqdm
import random
from mongo_utils import get_collection

collection = get_collection("chatbot_db", "new_messages")
data = list(collection.find())
for doc in data:
    doc.pop("_id", None)

messages_df = pd.DataFrame(data)
messages_df["question"] = messages_df["question"].fillna("")

def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

messages_df["sentiment"] = messages_df["question"].apply(get_sentiment)

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

messages_df["user_profile"] = messages_df["question"].apply(segment_user_profile)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
device = torch.device("cpu")
model.to(device)
model.eval()

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
messages_df["embedding"] = messages_df["question"].progress_apply(get_embedding)
embeddings = np.vstack(messages_df["embedding"].values)

print("Clustering questions...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
messages_df["cluster"] = kmeans.fit_predict(embeddings)

cluster_to_class = {0: "Informational", 1: "Conversational", 2: "Transactional"}

def refine_class(question, cluster):
    question = question.lower()
    if cluster == 0 and any(kw in question for kw in ["book", "order", "reserve"]):
        return "Transactional"
    elif cluster == 1 and any(kw in question for kw in ["tell", "what", "how"]):
        return "Informational"
    return cluster_to_class[cluster]

messages_df["question_type_nlp"] = messages_df.apply(lambda x: refine_class(x["question"], x["cluster"]), axis=1)

vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
tfidf_matrix = vectorizer.fit_transform(messages_df["question"])
topic_model = KMeans(n_clusters=5, random_state=42).fit(tfidf_matrix)
messages_df["topic_cluster"] = topic_model.labels_

nbq_lookup = defaultdict(list)
for _, row in messages_df.iterrows():
    nbq_lookup[row["question_type_nlp"]].append(row["question"])

def suggest_nbq(q_class):
    pool = nbq_lookup.get(q_class, [])
    return random.choice(pool) if pool else ""

messages_df["next_best_question"] = messages_df["question_type_nlp"].apply(suggest_nbq)
topic_ranking = messages_df["topic_cluster"].value_counts().to_dict()
messages_df["topic_rank_score"] = messages_df["topic_cluster"].map(topic_ranking)

output_path = "messages_classified_nlp_mongo.csv"
messages_df.to_csv(output_path, index=False, encoding="utf-8")
print(f"Enhanced classified data saved to {output_path}")
