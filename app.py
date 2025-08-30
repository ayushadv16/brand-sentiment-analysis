# -------------------------------------------------------------
# Streamlit Sentiment Dashboard (Kaggle-ready) with WordCloud
# -------------------------------------------------------------
import re
import io
import string
import unicodedata
import pandas as pd
import numpy as np
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Social Sentiment | Brand Management",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure VADER lexicon is available
try:
    _ = SentimentIntensityAnalyzer()
except Exception:
    nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Sidebar theme toggle
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
with st.sidebar:
    st.header("⚙️ Controls")
    st.session_state.dark_mode = st.toggle("Dark mode", value=st.session_state.dark_mode, help="Switch dashboard theme")

is_dark = st.session_state.dark_mode

# CSS with improved color contrast for light mode
PRIMARY_BG_DARK = "#0E1117"
PRIMARY_BG_LIGHT = "#FFFFFF"
TEXT_DARK = "#E6E6E6"
TEXT_LIGHT = "#111827"
CARD_DARK = "#161A23"
CARD_LIGHT = "#F8FAFC"
ACCENT = "#6C5CE7"

st.markdown(f"""
<style>
:root {{
    --bg-dark: {PRIMARY_BG_DARK};
    --fg-dark: {TEXT_DARK};
    --card-dark: {CARD_DARK};
    --bg-light: {PRIMARY_BG_LIGHT};
    --fg-light: {TEXT_LIGHT};
    --card-light: {CARD_LIGHT};
    --accent: {ACCENT};
}}
.stApp {{
    background: {"var(--bg-dark)" if is_dark else "var(--bg-light)"} !important;
    color: {"var(--fg-dark)" if is_dark else "var(--fg-light)"} !important;
}}
.block-container {{ padding-top: 2rem; }}
.card {{
    background: {"var(--card-dark)" if is_dark else "var(--card-light)"};
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0, {0.35 if is_dark else 0.08});
}}
.metric {{ display: grid; grid-template-columns: auto 1fr; gap: .25rem 1rem; align-items: center; }}
.metric .value {{ font-size: 1.8rem; font-weight: 700; }}
.metric .label {{ opacity: .9; }}
.accent {{ color: var(--accent); }}
.subtle {{ opacity: .8; }}
.footer {{ opacity: .7; font-size: .9rem; margin-top: .75rem; }}
</style>
""", unsafe_allow_html=True)

plotly_template = "plotly_dark" if is_dark else "plotly"

# Helper functions
TEXT_CANDIDATES = ["text", "content", "tweet", "clean_text", "Tweet", "message", "body", "review"]
LABEL_CANDIDATES = ["sentiment", "label", "airline_sentiment", "target", "polarity"]
TIME_CANDIDATES = ["created_at", "date", "timestamp", "time"]
BRAND_CANDIDATES = ["brand", "entity", "airline", "company", "topic"]
SENTIMENT_MAP = {0: "Negative", 1: "Neutral", 2: "Positive", -1: "Negative", 2.0: "Positive", 4: "Positive"}
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def pick_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    lower_cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_cols:
            return lower_cols[c]
    return None

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    txt_col = pick_first_column(df, TEXT_CANDIDATES)
    if txt_col is None:
        str_cols = [c for c in df.columns if df[c].dtype == object]
        txt_col = max(str_cols, key=lambda c: df[c].astype(str).str.len().mean()) if str_cols else df.columns[0]
    df.rename(columns={txt_col: "text"}, inplace=True)
    lab_col = pick_first_column(df, LABEL_CANDIDATES)
    if lab_col:
        df.rename(columns={lab_col: "sentiment"}, inplace=True)
        def map_lab(v):
            if pd.isna(v): return None
            if isinstance(v, (int, float)) and v in SENTIMENT_MAP: return SENTIMENT_MAP[v]
            vs = str(v).strip().lower()
            if vs in ("pos", "+", "positive"): return "Positive"
            if vs in ("neg", "-", "negative"): return "Negative"
            if vs in ("neu", "neutral"): return "Neutral"
            return v
        df["sentiment"] = df["sentiment"].map(map_lab)
    tcol = pick_first_column(df, TIME_CANDIDATES)
    if tcol:
        df.rename(columns={tcol: "created_at"}, inplace=True)
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    bcol = pick_first_column(df, BRAND_CANDIDATES)
    if bcol:
        df.rename(columns={bcol: "brand"}, inplace=True)
    keep = [c for c in ["text", "sentiment", "created_at", "brand"] if c in df.columns]
    id_like = [c for c in df.columns if c.lower() in ("id", "tweet_id", "id_str")] 
    other = [c for c in df.columns if c not in keep + id_like]
    cols = id_like + keep + other
    return df[cols]

def clean_text(s: str) -> str:
    if not isinstance(s, str): s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"@[A-Za-z0-9_]+", " ", s)
    s = re.sub(r"#[A-Za-z0-9_]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = s.translate(PUNCT_TABLE)
    return s.strip()

def vader_label(row_text: str) -> tuple[str, float]:
    scores = sia.polarity_scores(str(row_text))
    comp = scores["compound"]
    if comp > 0.05: return "Positive", comp
    if comp < -0.05: return "Negative", comp
    return "Neutral", comp

def top_ngrams(texts: list[str], ngram_range=(1,2), top_k=15) -> pd.DataFrame:
    vec = CountVectorizer(ngram_range=ngram_range, stop_words="english", min_df=2)
    X = vec.fit_transform(texts)
    freqs = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array(list(vec.vocabulary_.keys()))
    items = sorted(zip(vocab, freqs), key=lambda x: x[1], reverse=True)[:top_k]
    return pd.DataFrame(items, columns=["term", "count"])

# Sidebar: Data Input
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Data Source")
opt = st.sidebar.radio(
    "Choose input method",
    ["Upload CSV (Kaggle)", "Paste small sample"],
    help="Tip: Download a CSV from Kaggle and upload it here."
)
if opt == "Upload CSV (Kaggle)":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])    
else:
    sample_text = st.sidebar.text_area(
        "Paste a few lines (one post per line)", height=150,
        placeholder="I love BrandX!\nBrandX customer service is terrible.\nBrandX is okay, nothing special."
    )
    uploaded = io.BytesIO(pd.DataFrame({"text": sample_text.splitlines()}).to_csv(index=False).encode("utf-8")) if sample_text else None
min_rows = st.sidebar.slider("Minimum rows to analyze", 50, 10000, 200, 50)

# Title
st.markdown("# 📊 Social Media Sentiment — Brand Management Dashboard")
st.caption("Upload a Kaggle dataset of tweets/posts to get live sentiment, trending topics, and actionable insights.")
if not uploaded:
    st.stop()

# Load & normalize CSV
@st.cache_data(show_spinner=False)
def load_any_csv(file_like) -> pd.DataFrame:
    try:
        return pd.read_csv(file_like)
    except UnicodeDecodeError:
        file_like.seek(0)
        return pd.read_csv(file_like, encoding="latin-1")

with st.spinner("Reading CSV…"):
    raw = load_any_csv(uploaded)
    df = normalize_dataframe(raw)

if len(df) < min_rows:
    st.warning(f"Dataset has only {len(df)} rows (< {min_rows}). Add more data for reliable insights.")

# Clean & compute sentiment
with st.spinner("Cleaning & scoring sentiment…"):
    df["text_clean"] = df["text"].astype(str).map(clean_text)
    if "sentiment" not in df.columns or df["sentiment"].isna().all():
        labs, comps = zip(*df["text"].map(vader_label))
        df["sentiment"] = labs
        df["compound"] = comps
    else:
        df["compound"] = df["text"].map(lambda t: sia.polarity_scores(str(t))["compound"])

# KPIs
pos = int((df["sentiment"] == "Positive").sum())
neg = int((df["sentiment"] == "Negative").sum())
neu = int((df["sentiment"] == "Neutral").sum())
net_score = round(df["compound"].mean(), 3)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='card metric'><div class='label'>Total Posts</div><div class='value'>{len(df):,}</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='card metric'><div class='label'>Positive</div><div class='value accent'>{pos:,}</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='card metric'><div class='label'>Negative</div><div class='value accent'>{neg:,}</div></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='card metric'><div class='label'>Net Sentiment</div><div class='value'>{net_score}</div></div>", unsafe_allow_html=True)

# Sentiment analytics
st.subheader("📈 Sentiment Analytics")
dist = df["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
grp = df.groupby("sentiment")["compound"].mean().reset_index().sort_values("compound")
trend_fig = None
if "created_at" in df.columns and df["created_at"].notna().any():
    trend = df.dropna(subset=["created_at"]).copy()
    trend = trend.set_index("created_at").resample("D").agg({"compound":"mean", "text":"count"}).rename(columns={"text":"posts"}).reset_index()
    trend_fig = px.line(trend, x="created_at", y="compound", template=plotly_template, markers=True)
    trend_fig.update_layout(height=400, yaxis_title="Mean Compound", xaxis_title="Date",
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")

color_map = {
    "Positive": "#27AE60" if not is_dark else "#6FCF97",  # lighter green for dark mode
    "Negative": "#EB5757" if not is_dark else "#FF6B6B",  # lighter red for dark mode
    "Neutral": "#F2C94C" if not is_dark else "#FAD961"   # lighter yellow for dark mode
}

text_color = "white" if is_dark else "black"
border_color = "white" if is_dark else "black"

# Sentiment Distribution Bar Chart
dist_fig = px.bar(
    dist,
    x="sentiment",
    y="count",
    text="count",
    template=plotly_template,
    color="sentiment",
    color_discrete_map=color_map,
)
dist_fig.update_traces(
    textposition="outside",
    textfont_color=text_color,
    marker_line_color=border_color,
    marker_line_width=1.5
)
dist_fig.update_layout(
    height=380,
    xaxis_title="",
    yaxis_title="Posts",
    xaxis=dict(
        tickfont=dict(size=13, color=text_color, family="Arial"),
        showgrid=True,
        zeroline=False,
    ),
    yaxis=dict(
        tickfont=dict(size=13, color=text_color, family="Arial"),
        showgrid=True,
        zeroline=False,
    ),
    legend=dict(
        title=None,
        font=dict(size=14, color=text_color, family="Arial"),
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
    ),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)

# Average Polarity Bar Chart
polarity_fig = px.bar(
    grp,
    x="sentiment",
    y="compound",
    text="compound",
    template=plotly_template,
    color="sentiment",
    color_discrete_map=color_map,
)
polarity_fig.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside",
    textfont_color=text_color,
    marker_line_color=border_color,
    marker_line_width=1.5
)
polarity_fig.update_layout(
    height=380,
    xaxis_title="",
    yaxis_title="Avg Compound",
    xaxis=dict(
        tickfont=dict(size=13, color=text_color, family="Arial"),
        showgrid=True,
        zeroline=False,
    ),
    yaxis=dict(
        tickfont=dict(size=13, color=text_color, family="Arial"),
        showgrid=True,
        zeroline=False,
    ),
    legend=dict(
        title=None,
        font=dict(size=14, color=text_color, family="Arial"),
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
    ),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Sentiment Distribution**")
    st.plotly_chart(dist_fig, use_container_width=True)
with col2:
    st.markdown("**Average Polarity by Sentiment**")
    st.plotly_chart(polarity_fig, use_container_width=True)

if trend_fig:
    st.markdown("**Sentiment Trend Over Time**")
    st.plotly_chart(trend_fig, use_container_width=True)

# Trending topics & keywords
st.subheader("Trending Topics & Keywords")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("**Overall**")
    overall = top_ngrams(df["text_clean"].tolist(), (1,2), 15)
    st.dataframe(overall, use_container_width=True, hide_index=True)
with col_b:
    st.markdown("**Positive**")
    pos_df = top_ngrams(df.query("sentiment=='Positive'")["text_clean"].tolist() or [""], (1,2), 15)
    st.dataframe(pos_df, use_container_width=True, hide_index=True)
with col_c:
    st.markdown("**Negative**")
    neg_df = top_ngrams(df.query("sentiment=='Negative'")["text_clean"].tolist() or [""], (1,2), 15)
    st.dataframe(neg_df, use_container_width=True, hide_index=True)

# WordCloud
st.subheader("🖼️ WordCloud of Trending Words")
def generate_wordcloud(texts, is_dark):
    combined = " ".join(texts)
    if not combined.strip():
        return None
    wc = WordCloud(
        width=600, height=400,
        background_color="#0E1117" if is_dark else "white",
        colormap="Pastel1" if is_dark else "tab10",
        stopwords=None,
        max_words=150
    ).generate(combined)
    return wc

col_wc1, col_wc2, col_wc3 = st.columns(3)
with col_wc1:
    st.markdown("**Overall**")
    wc = generate_wordcloud(df["text_clean"].tolist(), is_dark)
    if wc:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.caption("No text available")

with col_wc2:
    st.markdown("**Positive**")
    wc = generate_wordcloud(df.query("sentiment=='Positive'")["text_clean"].tolist(), is_dark)
    if wc:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.caption("No positive posts found")

with col_wc3:
    st.markdown("**Negative**")
    wc = generate_wordcloud(df.query("sentiment=='Negative'")["text_clean"].tolist(), is_dark)
    if wc:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.caption("No negative posts found")

# Brand drill-down
if "brand" in df.columns and df["brand"].notna().any():
    st.subheader("Brand Drill-down")
    brands = [b for b in df["brand"].astype(str).unique() if b and b.lower() != "nan"]
    pick = st.selectbox("Choose brand", options=sorted(brands))
    sub = df[df["brand"].astype(str) == str(pick)]
    dist_b = sub["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
    figb = px.bar(dist_b, x="sentiment", y="count", text="count", template=plotly_template)
    figb.update_traces(textposition="outside")
    figb.update_layout(height=360, xaxis_title="", yaxis_title="Posts")
    st.plotly_chart(figb, use_container_width=True)

# Data preview & export
with st.expander("Preview analyzed data"):
    st.dataframe(df.head(200), use_container_width=True)

@st.cache_data
def to_csv_bytes(_df: pd.DataFrame) -> bytes:
    return _df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download analyzed CSV",
    data=to_csv_bytes(df[[c for c in ["text","sentiment","compound","created_at","brand","text_clean"] if c in df.columns]]),
    file_name="sentiment_analyzed.csv",
    mime="text/csv",
)

# Quick insights narrative
st.subheader("Insights for Brand Management")
insights = []
if neg > pos:
    insights.append("Negative chatter outweighs positive — prioritize service recovery and investigate top negative keywords above.")
else:
    insights.append("Positive chatter outweighs negative — amplify what's working via campaigns and testimonials.")

if net_score < -0.05:
    insights.append("Overall sentiment is unfavorable (compound < 0). Consider immediate response playbooks and FAQ updates.")
elif net_score > 0.2:
    insights.append("Overall sentiment is favorable. Maintain momentum with influencer outreach and product highlights.")
else:
    insights.append("Neutral overall sentiment. Opportunity to differentiate with proactive engagement.")

st.markdown(f"<div class='card'>" + "".join([f"<p>• {msg}</p>" for msg in insights]) + "</div>", unsafe_allow_html=True)

examples = {
    "Positive": df[df["sentiment"]=="Positive"].sort_values("compound", ascending=False).head(5),
    "Negative": df[df["sentiment"]=="Negative"].sort_values("compound").head(5),
}

ex_col1, ex_col2 = st.columns(2)
with ex_col1:
    st.markdown("**Top Positive Mentions**")
    if len(examples["Positive"]) == 0: st.caption("No positive posts found.")
    else: 
        for i, row in examples["Positive"].iterrows():
            st.write(f"✅ {row['text']}")

with ex_col2:
    st.markdown("**Top Negative Mentions**")
    if len(examples["Negative"]) == 0: st.caption("No negative posts found.")
    else: 
        for i, row in examples["Negative"].iterrows():
            st.write(f"⚠️ {row['text']}")

st.markdown("\n")

st.markdown("<br><br><br>", unsafe_allow_html=True)
st.caption(" Made with ❤️ By Ayush Patil 25306A1026 — &copy; 2025")

