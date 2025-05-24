# 🧠 GenAI Chatbot Evaluation Toolkit

This project provides a set of tools to analyze and evaluate user interactions with a Generative AI (GenAI) chatbot. It supports question classification, sentiment analysis, topic modeling, response relevance evaluation, and success/failure labeling based on interaction quality.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `success_ratio.py` | Evaluates response relevance and completeness using semantic similarity and linguistic checks, then computes success/failure metrics. |
| `relevance_evaluation.py` | Computes cosine similarity between questions and answers to label their semantic relevance. |
| `message_classification_toolkit.py` | Performs NLP-based classification of user questions into types (e.g., informational, transactional) with sentiment analysis and clustering. |
| `message_classification_toolkit_mcp.py` | Enhances classification using zero-shot learning (`facebook/bart-large-mnli`) and adds next-best-question suggestions and topic ranking. |
| `Assignment.docx` | Describes the objective: evaluating GenAI chatbot interactions using classification, sentiment, topic modeling, and success metrics. |

---

## 🧪 Features

### ✅ **Success-Failure Analysis**
- Cosine similarity (DistilBERT) to measure relevance.
- Linguistic checks for completeness (SVO structure, vague phrases).
- Assigns interaction outcome: success or failure.

### 💬 **Message Classification**
- Classifies questions into types:
  - **Informational**
  - **Transactional**
  - **Conversational**
- Clusters topics using TF-IDF + KMeans.
- Segments user profile: Curious, Action-Oriented, Casual, etc.
- Computes sentiment using TextBlob.
- Suggests Next Best Question for each class.

### 📈 **Relevance Evaluation**
- Uses DistilBERT embeddings to compute similarity scores.
- Labels answers as "Relevant", "Somewhat Relevant", or "Irrelevant".

---

## 🚀 How to Use

### 1. Prepare Your Data
Your CSV must contain:
```csv
question,response
"What is AI?", "AI stands for Artificial Intelligence..."
```

### 2. Run Evaluation Scripts

#### Success & Failure Metrics:
```bash
python success_ratio.py
```

#### Relevance Evaluation:
```bash
python relevance_evaluation.py
```

#### NLP Classification:
```bash
python message_classification_toolkit.py
```

#### Zero-Shot Enhanced Classification:
```bash
python message_classification_toolkit_mcp.py
```

---

## 📦 Requirements

Install the required Python packages:
```bash
pip install pandas numpy torch transformers spacy textblob scikit-learn tqdm
python -m spacy download en_core_web_sm
```

---

## 🧾 Output Files

- `messages_with_success_failure.csv` – Includes relevance, completeness, and success labels.
- `messages_with_relevance.csv` – Relevance labels only.
- `messages_classified_nlp.csv` – NLP-based classification with clustering and sentiment.
- `messages_classified_zero_shot.csv` – Zero-shot classification output.

---

## 🧠 Objective Summary (from `Assignment.docx`)

The core goals are:
- Set up data access from MongoDB/chat logs.
- Categorize questions and segment user profiles.
- Evaluate the relevance and completeness of answers.
- Determine chatbot interaction success ratio.
- Rank topics and suggest personalized follow-ups.