import re
import numpy as np
import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

# Page config
st.set_page_config(
    page_title="Personalized Reading Assistant",
    page_icon="📰",
    layout="wide",
)


# Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0F172A;
        color: #E5E7EB;
    }

    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding-top: 2.5rem;
        max-width: 900px;
    }

    /* ---------- Hero ---------- */
    .eyebrow {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        color: #22C55E;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .hero-title {
        font-family: 'Sora', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        color: #E5E7EB;
        line-height: 1.15;
        margin-bottom: 6px;
    }
    .hero-title span {
        color: #3B82F6;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #94A3B8;
        margin-bottom: 2rem;
        max-width: 620px;
        line-height: 1.55;
    }

    /* ---------- Control panel ---------- */
    .panel {
        background-color: #1E293B;
        border: 1px solid #2B3A52;
        border-radius: 14px;
        padding: 22px 24px 6px 24px;
        margin-bottom: 28px;
    }
    .panel-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 10px;
    }

    div[data-testid="stTextInput"] input {
        background-color: #0F172A;
        color: #E5E7EB;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.95rem;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 1px #3B82F6;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: #64748B;
    }

    div[role="radiogroup"] label {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 6px 14px;
        margin-right: 8px;
        color: #CBD5E1 !important;
    }

    .stSlider label {
        color: #CBD5E1 !important;
    }

    .stButton > button {
        background-color: #3B82F6;
        color: #0F172A;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.4rem;
        font-size: 0.95rem;
        transition: background-color 0.15s ease;
    }
    .stButton > button:hover {
        background-color: #22C55E;
        color: #0F172A;
    }

    .stCaption, .stMarkdown p em {
        color: #64748B !important;
    }

    /* ---------- Result cards ---------- */
    .results-heading {
        font-family: 'Sora', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #E5E7EB;
        margin: 4px 0 18px 0;
        border-bottom: 1px solid #2B3A52;
        padding-bottom: 10px;
    }

    .card {
        background-color: #1E293B;
        border: 1px solid #2B3A52;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        display: flex;
        gap: 16px;
        align-items: flex-start;
    }
    .rank-badge {
        font-family: 'Sora', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #334155;
        background-color: #0F172A;
        border: 1px solid #2B3A52;
        border-radius: 8px;
        min-width: 38px;
        height: 38px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .card-body { width: 100%; }
    .card-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .category-badge {
        display: inline-block;
        background-color: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.35);
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .score-value {
        font-size: 0.8rem;
        font-weight: 600;
        color: #22C55E;
        font-variant-numeric: tabular-nums;
    }
    .score-bar-track {
        background-color: #0F172A;
        border-radius: 4px;
        height: 5px;
        width: 100%;
        margin: 8px 0 10px 0;
        overflow: hidden;
    }
    .score-bar-fill {
        background-color: #22C55E;
        height: 100%;
        border-radius: 4px;
    }
    .snippet-text {
        color: #94A3B8;
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 0;
    }

    .footer-note {
        text-align: center;
        color: #475569;
        font-size: 0.8rem;
        margin-top: 40px;
        padding-top: 18px;
        border-top: 1px solid #1E293B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# NLTK setup
@st.cache_resource
def setup_nltk():
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

setup_nltk()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)


# Data + model loading
@st.cache_data
def load_data():
    df = pd.read_csv("bbc-text.csv")
    df["clean_text"] = df["text"].apply(preprocess_text)
    return df


@st.cache_resource
def build_tfidf(df):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(df["clean_text"])
    return vectorizer, matrix


@st.cache_resource
def build_word2vec(df):
    tokenized_docs = [text.split() for text in df["clean_text"]]

    model_cbow = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, sg=0)
    model_sg = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, sg=1)

    def get_avg_embedding(tokens, model, size=100):
        vecs = [model.wv[w] for w in tokens if w in model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(size)

    vecs_cbow = np.array([get_avg_embedding(t, model_cbow) for t in tokenized_docs])
    vecs_sg = np.array([get_avg_embedding(t, model_sg) for t in tokenized_docs])

    return model_cbow, model_sg, vecs_cbow, vecs_sg, get_avg_embedding


# Recommendation functions
def recommend_tfidf(query, vectorizer, matrix, df, top_n=5):
    query_clean = preprocess_text(query)
    query_vec = vectorizer.transform([query_clean])
    scores = cosine_similarity(query_vec, matrix).flatten()
    top_idx = scores.argsort()[-top_n:][::-1]
    results = df.iloc[top_idx].copy()
    results["similarity_score"] = scores[top_idx]
    return results


def recommend_word2vec(query, model, doc_vectors, df, get_avg_embedding, top_n=5):
    query_tokens = preprocess_text(query).split()
    query_vec = get_avg_embedding(query_tokens, model).reshape(1, -1)
    scores = cosine_similarity(query_vec, doc_vectors).flatten()
    top_idx = scores.argsort()[-top_n:][::-1]
    results = df.iloc[top_idx].copy()
    results["similarity_score"] = scores[top_idx]
    return results


def show_results(results):
    max_score = max(results["similarity_score"].max(), 1e-6)
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        snippet = row["text"][:220].strip() + "..."
        bar_pct = max(6, round((row["similarity_score"] / max_score) * 100))
        st.markdown(
            f"""
            <div class="card">
                <div class="rank-badge">{rank:02d}</div>
                <div class="card-body">
                    <div class="card-top-row">
                        <span class="category-badge">{row['category']}</span>
                        <span class="score-value">{row['similarity_score']:.3f}</span>
                    </div>
                    <div class="score-bar-track">
                        <div class="score-bar-fill" style="width:{bar_pct}%;"></div>
                    </div>
                    <p class="snippet-text">{snippet}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Main app
df = load_data()
tfidf_vectorizer, tfidf_matrix = build_tfidf(df)
model_cbow, model_sg, vecs_cbow, vecs_sg, get_avg_embedding = build_word2vec(df)

st.markdown('<div class="eyebrow">NLP · TF-IDF · Word2Vec</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-title">Personalized <span>Reading</span> Assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="hero-subtitle">Describe what you feel like reading, and get the closest '
    f'matching articles from {len(df)} BBC News stories across {df["category"].nunique()} '
    f'categories — ranked by two different recommendation engines.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-label">Recommendation engine</div>', unsafe_allow_html=True)
method = st.radio(
    "Recommendation method",
    ["TF-IDF", "Word2Vec · CBOW", "Word2Vec · Skip-gram"],
    horizontal=True,
    label_visibility="collapsed",
)
top_n = st.slider("Number of results", 3, 10, 5)
query = st.text_input(
    "Query",
    placeholder="e.g. the future of mobile technology and digital gadgets",
    label_visibility="collapsed",
)
run = st.button("Get Recommendations", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

if run and query.strip():
    with st.spinner("Finding the best matches..."):
        if method == "TF-IDF":
            results = recommend_tfidf(query, tfidf_vectorizer, tfidf_matrix, df, top_n)
        elif method == "Word2Vec · CBOW":
            results = recommend_word2vec(query, model_cbow, vecs_cbow, df, get_avg_embedding, top_n)
        else:
            results = recommend_word2vec(query, model_sg, vecs_sg, df, get_avg_embedding, top_n)

    st.markdown(
        f'<div class="results-heading">Top {top_n} matches · {method}</div>',
        unsafe_allow_html=True,
    )
    show_results(results)
elif run:
    st.warning("Type a query first — tell the assistant what you'd like to read about.")

st.markdown(
    '<div class="footer-note">Built with TF-IDF &amp; Word2Vec (CBOW / Skip-gram) '
    '&middot; BBC News dataset</div>',
    unsafe_allow_html=True,
)