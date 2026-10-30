import io
import re
import html
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from nltk.tokenize import TweetTokenizer

st.set_page_config(
    page_title="Pulse | Social Sentiment Intelligence",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

LABELS = ["negative", "neutral", "positive"]
TOKENIZER = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {background: rgba(120,120,120,.07); padding: 14px; border-radius: 12px;}
    .hero {padding: 1.25rem 1.4rem; border: 1px solid rgba(128,128,128,.25); border-radius: 16px;
           background: linear-gradient(115deg, rgba(59,130,246,.12), rgba(16,185,129,.07));}
    .muted {color: #8b929c; font-size: .9rem;}
    div[data-testid="stDataFrame"] {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

def clean_tweet(text):
    """Normalize tweet text while retaining hashtags, emojis and useful negation words."""
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"(?<!\w)@\w+", " USER ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = TOKENIZER.tokenize(text)
    return " ".join(tokens)

def normalize_label(value):
    v = str(value).strip().lower()
    mapping = {
        "0": "negative", "negative": "negative", "neg": "negative",
        "-1": "negative", "1": "neutral", "neutral": "neutral", "neu": "neutral",
        "2": "positive", "positive": "positive", "pos": "positive",
        "4": "positive"
    }
    return mapping.get(v)

@st.cache_data(show_spinner=False)
def load_demo():
    rows = [
        ("The new update is fantastic, everything feels faster!", "positive"),
        ("Absolutely love this feature. Great work team!", "positive"),
        ("Customer support solved my issue quickly. Thank you!", "positive"),
        ("This is the best experience I have had in months.", "positive"),
        ("Finally, a product that actually delivers. #happy", "positive"),
        ("The launch event was brilliant and full of energy.", "positive"),
        ("Impressed by the quality and attention to detail.", "positive"),
        ("That was a wonderful performance. Loved every minute.", "positive"),
        ("Fast delivery and the packaging was excellent.", "positive"),
        ("Huge improvement over the previous version.", "positive"),
        ("The team did an amazing job on this release.", "positive"),
        ("Works perfectly for my daily workflow.", "positive"),
        ("I am really pleased with the results.", "positive"),
        ("Such a smooth experience from start to finish.", "positive"),
        ("This made my day. Highly recommended!", "positive"),
        ("Great value for money and reliable service.", "positive"),
        ("The new design looks clean and beautiful.", "positive"),
        ("Excellent response time from the support team.", "positive"),
        ("Everything arrived on time and works as expected.", "positive"),
        ("A solid release. Looking forward to what comes next.", "positive"),
        ("The app keeps crashing and I have lost my work.", "negative"),
        ("Very disappointed with the service. Nobody replied.", "negative"),
        ("This update made everything slower and harder to use.", "negative"),
        ("Terrible experience. I want a refund.", "negative"),
        ("The product stopped working after two days.", "negative"),
        ("What a waste of money. Do not buy this.", "negative"),
        ("Support was rude and did not solve anything.", "negative"),
        ("The delivery is late again. Completely unacceptable.", "negative"),
        ("I hate the new interface. It is confusing.", "negative"),
        ("The quality is poor and the price is too high.", "negative"),
        ("Another bug after the latest release. Frustrating.", "negative"),
        ("Nothing works as advertised. Very unhappy.", "negative"),
        ("The website is down and customers cannot log in.", "negative"),
        ("I have been charged twice. This is ridiculous.", "negative"),
        ("Worst customer experience I have had in years.", "negative"),
        ("The battery drains quickly and the phone overheats.", "negative"),
        ("This feature is broken. Please fix it.", "negative"),
        ("Waiting for hours with no update is unacceptable.", "negative"),
        ("The new policy is confusing and unfair.", "negative"),
        ("I regret purchasing this product.", "negative"),
        ("The company announced a product launch today.", "neutral"),
        ("The meeting is scheduled for 3 PM tomorrow.", "neutral"),
        ("The app is available on Android and iOS.", "neutral"),
        ("The team shared the quarterly report this morning.", "neutral"),
        ("There is a maintenance window this weekend.", "neutral"),
        ("The store opens at 9 AM on weekdays.", "neutral"),
        ("The conference will take place in Mumbai next month.", "neutral"),
        ("The latest version is 4.2.1.", "neutral"),
        ("A new branch is opening near the station.", "neutral"),
        ("The company posted its financial results today.", "neutral"),
        ("The package includes a charger and a user guide.", "neutral"),
        ("The event starts with a keynote presentation.", "neutral"),
        ("The survey is open until Friday.", "neutral"),
        ("The service supports multiple payment methods.", "neutral"),
        ("The train is expected to arrive at platform two.", "neutral"),
        ("The report contains data from the last quarter.", "neutral"),
        ("The website lists the updated office timings.", "neutral"),
        ("The announcement was made during the press briefing.", "neutral"),
        ("The new policy takes effect from 1 January.", "neutral"),
        ("The video has been uploaded to the official channel.", "neutral"),
    ]
    return pd.DataFrame(rows, columns=["text", "sentiment"])

def prepare_dataset(df, text_col, label_col):
    out = df[[text_col, label_col]].copy()
    out.columns = ["text", "sentiment"]
    out = out.dropna(subset=["text", "sentiment"])
    out["text"] = out["text"].astype(str).str.strip()
    out["sentiment"] = out["sentiment"].map(normalize_label)
    out = out[(out["text"] != "") & out["sentiment"].isin(LABELS)]
    out["clean_text"] = out["text"].map(clean_tweet)
    out = out[out["clean_text"].str.strip() != ""]
    out = out.drop_duplicates(subset=["clean_text"])
    return out.reset_index(drop=True)

def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, max_features=50000,
            sublinear_tf=True, strip_accents="unicode"
        )),
        ("classifier", LogisticRegression(
            max_iter=1500, class_weight="balanced", C=3.0,
            solver="lbfgs"
        ))
    ])

def get_upload():
    st.sidebar.markdown("### Data source")
    source = st.sidebar.radio("Choose input", ["Built-in demo dataset", "Upload labeled CSV"])
    if source == "Built-in demo dataset":
        return prepare_dataset(load_demo(), "text", "sentiment"), "Demo dataset"
    upload = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if upload is None:
        st.info("Upload a labeled CSV with a tweet/text column and a sentiment/label column.")
        st.stop()
    try:
        df = pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
    st.sidebar.caption("Choose the columns in the dataset.")
    text_col = st.sidebar.selectbox("Tweet text column", list(df.columns),
                                    index=0)
    possible = [c for c in df.columns if c != text_col]
    if not possible:
        st.error("CSV needs a separate sentiment label column for model training.")
        st.stop()
    label_col = st.sidebar.selectbox("Sentiment label column", possible,
                                     index=0)
    return prepare_dataset(df, text_col, label_col), upload.name

st.markdown("""
<div class="hero">
  <div class="muted">SOCIAL LISTENING • NLP • MACHINE LEARNING</div>
  <h1 style="margin:.25rem 0;">◉ PULSE — Social Sentiment Intelligence</h1>
  <div>Classify public-post text, evaluate model quality, inspect sentiment drivers and export scored datasets.</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Workspace")
    test_size = st.slider("Test data (%)", 10, 35, 20, 5)
    random_seed = st.number_input("Reproducibility seed", min_value=1, max_value=9999, value=42)
    st.caption("This prototype uses a stratified random split. For production evaluation, use time-based and group-aware validation where appropriate.")

dataset, source_name = get_upload()
if len(dataset) < 15:
    st.error("At least 15 valid labeled tweets are required. Add more examples for a meaningful evaluation.")
    st.stop()

counts = dataset["sentiment"].value_counts()
missing_classes = [x for x in LABELS if counts.get(x, 0) < 2]
if missing_classes:
    st.error("Need at least two examples in each class. Missing/underrepresented: " + ", ".join(missing_classes))
    st.stop()

X = dataset["clean_text"]
y = dataset["sentiment"]
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=int(random_seed),
        stratify=y
    )
    model = build_pipeline()
    with st.spinner("Training NLP model and evaluating holdout data..."):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.stop()

accuracy = accuracy_score(y_test, predictions)
macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    y_test, predictions, average="macro", zero_division=0
)
top_label = counts.idxmax()
total = len(dataset)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Labeled tweets", f"{total:,}")
k2.metric("Holdout accuracy", f"{accuracy:.1%}")
k3.metric("Macro F1", f"{macro_f1:.3f}")
k4.metric("Most common class", top_label.title())

tab1, tab2, tab3, tab4 = st.tabs([
    "Live analyzer", "Model performance", "Dataset insights", "Export & governance"
])

with tab1:
    st.subheader("Analyze a post")
    st.caption("Enter one post or upload a CSV in the batch section below.")
    sample_text = st.text_area(
        "Post text", height=110,
        placeholder="Example: The latest update is fast and easy to use!"
    )
    if st.button("Analyze sentiment", type="primary", use_container_width=False):
        if not sample_text.strip():
            st.warning("Enter some text first.")
        else:
            clean = clean_tweet(sample_text)
            probs = model.predict_proba([clean])[0]
            order = np.argsort(probs)[::-1]
            pred = model.classes_[order[0]]
            confidence = float(probs[order[0]])
            color = {"positive": "success", "neutral": "info", "negative": "error"}[pred]
            st.markdown(f"### Prediction: :{color}[{pred.upper()}]")
            a, b = st.columns(2)
            a.metric("Model confidence", f"{confidence:.1%}")
            b.metric("Runner-up", f"{model.classes_[order[1]].title()} ({probs[order[1]]:.1%})")
            prob_df = pd.DataFrame({"Sentiment": model.classes_, "Probability": probs})
            fig = px.bar(prob_df, x="Sentiment", y="Probability", range_y=[0, 1],
                         title="Class probability distribution", text_auto=".1%")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Confidence is the model's estimated class probability, not a guarantee of correctness.")

    st.divider()
    st.subheader("Batch analysis")
    batch_file = st.file_uploader("Upload CSV to score (no labels required)", type=["csv"], key="batch")
    if batch_file is not None:
        try:
            batch = pd.read_csv(batch_file)
            text_candidates = [c for c in batch.columns if c.lower() in
                               ["text", "tweet", "tweet_text", "content", "post"]]
            chosen_text = st.selectbox("Column containing posts", list(batch.columns),
                                       index=list(batch.columns).index(text_candidates[0])
                                       if text_candidates else 0)
            batch["predicted_sentiment"] = model.predict(batch[chosen_text].fillna("").map(clean_tweet))
            batch_probs = model.predict_proba(batch[chosen_text].fillna("").map(clean_tweet))
            batch["confidence"] = batch_probs.max(axis=1).round(4)
            st.dataframe(batch.head(100), use_container_width=True)
            st.download_button(
                "Download scored CSV",
                data=batch.to_csv(index=False).encode("utf-8"),
                file_name="pulse_scored_posts.csv",
                mime="text/csv"
            )
        except Exception as exc:
            st.error(f"Batch scoring failed: {exc}")

with tab2:
    st.subheader("Model performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("Macro precision", f"{macro_p:.3f}")
    m2.metric("Macro recall", f"{macro_r:.3f}")
    m3.metric("Macro F1-score", f"{macro_f1:.3f}")
    st.caption("Metrics are calculated on a held-out subset. Small or unrepresentative datasets can produce misleading scores.")
    report = classification_report(y_test, predictions, labels=LABELS,
                                   output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
    cm = confusion_matrix(y_test, predictions, labels=LABELS)
    fig = px.imshow(cm, x=[x.title() for x in LABELS], y=[x.title() for x in LABELS],
                    text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    title="Confusion matrix")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Reading the matrix:** diagonal cells are correct predictions; off-diagonal cells show which sentiments the model confuses.")

with tab3:
    st.subheader("Dataset composition")
    left, right = st.columns(2)
    with left:
        dist = dataset["sentiment"].value_counts().reindex(LABELS, fill_value=0).rename_axis("sentiment").reset_index(name="count")
        fig = px.pie(dist, names="sentiment", values="count", hole=.55,
                     title="Sentiment class distribution")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        dataset["word_count"] = dataset["text"].str.split().str.len()
        fig = px.histogram(dataset, x="word_count", color="sentiment",
                           nbins=25, title="Post length distribution")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Data quality checks**")
    quality = pd.DataFrame({
        "Check": ["Rows after cleaning", "Duplicate posts removed", "Empty/invalid rows excluded",
                  "Negative examples", "Neutral examples", "Positive examples"],
        "Result": [len(dataset), "See source cleaning", "See source cleaning",
                   int(counts.get("negative", 0)), int(counts.get("neutral", 0)),
                   int(counts.get("positive", 0))]
    })
    st.dataframe(quality, hide_index=True, use_container_width=True)
    st.caption("The demo dataset is hand-written for demonstrating the workflow. It is not a representative sample of real X/Twitter users.")

with tab4:
    st.subheader("Export and responsible use")
    st.write("The model is trained during this app session. It does not send uploaded text to an external sentiment API.")
    st.write("Avoid uploading private messages, personal identifiers or data you are not authorized to process.")
    st.write("Do not use sentiment scores alone for employee, customer or individual-level consequential decisions.")
    st.write("This application does not connect to X/Twitter or collect live posts. Import data you have permission to use.")
    export = dataset[["text", "sentiment"]].copy()
    st.download_button("Download cleaned training dataset",
                       data=export.to_csv(index=False).encode("utf-8"),
                       file_name="pulse_cleaned_training_data.csv",
                       mime="text/csv")
    st.markdown("**Model card**")
    st.markdown("""
    - **Task:** three-class text sentiment classification.
    - **Algorithm:** TF-IDF word unigrams/bigrams + Logistic Regression.
    - **Preprocessing:** NLTK TweetTokenizer; URL and mention normalization.
    - **Labels:** negative, neutral, positive.
    - **Known risks:** sarcasm, slang, code-switching, domain shift, bots, class imbalance and ambiguous language.
    - **Not supported:** reliable multilingual sentiment, emotion detection, bot detection or causal explanation.
    """)

st.markdown("---")
st.markdown(
    "<div class='muted'>PULSE • Prototype for learning and analyst workflow • "
    f"Session generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</div>",
    unsafe_allow_html=True
)
