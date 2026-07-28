# 📰 Personalized Reading Assistant
Comparing TF-IDF and Word2Vec for content-based article recommendation — given a query, find the most relevant news articles using two very different ways of representing text: sparse keyword vectors vs. dense semantic embeddings.

## 🎯 The Problem
Recommending relevant articles based on a user query is a core NLP task behind every "you might also like" feed. But there's no single best way to represent text for similarity search — a **sparse** method like TF-IDF matches exact words, while a **dense** method like Word2Vec tries to understand meaning.

## 🧪 What I Did
1. **Preprocessed the Text** — Cleaned 2,225 BBC News articles (removed URLs/special characters, lowercased, tokenized, removed stopwords, lemmatized) and explored the corpus with unigram/bigram/trigram frequency plots and a word cloud.
2. **Built a TF-IDF Recommender** — Vectorized articles with TF-IDF and used cosine similarity to rank articles against a user's query.
3. **Built a Word2Vec Recommender** — Trained both CBOW and Skip-gram embeddings from scratch on the corpus, represented each article as the average of its word vectors, and ranked articles the same way.
4. **Compared the Two Approaches** — Ran both methods on the same queries and analyzed where each one succeeds or breaks down.
5. **Built an Interactive Demo** — Wrapped both recommenders in a Streamlit app so anyone can type a query and get live recommendations from either method.

## 📊 Results

| Method | Best At | Main Weakness |
|---|---|---|
| **TF-IDF** | Exact keyword matching — reliably surfaces articles sharing specific query terms | Treats synonyms as unrelated (e.g. "phone" vs. "mobile"); ignores word order |
| **Word2Vec (CBOW)** | Fast to train, works well on frequent words | Averaging vectors can dilute the importance of one strong keyword |
| **Word2Vec (Skip-gram)** | Better at rare words and deeper contextual relationships | Needs more data to train well; slower than CBOW |

For this dataset, **TF-IDF gave the clearest, most directly relevant results**, since news headlines tend to rely on specific keywords. Word2Vec was "smarter" in the sense that it could surface topically related articles even without shared vocabulary — but that same flexibility sometimes made its recommendations less precise. There's no absolute winner here: it depends on whether the goal is precise keyword retrieval or broader topic discovery.

## 🛠️ Tech Stack
scikit-learn (TF-IDF, cosine similarity) · gensim (Word2Vec) · NLTK (preprocessing) · pandas · Streamlit

## 🚀 Try It Yourself
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook article_recommendation_system.ipynb
```

Or launch the interactive app:
```bash
streamlit run app.py
```
