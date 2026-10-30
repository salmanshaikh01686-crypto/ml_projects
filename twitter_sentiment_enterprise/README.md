# PULSE — Social Sentiment Intelligence

A professional, local-first Streamlit application for three-class sentiment analysis of post/tweet text.

## What makes this different from a basic sentiment notebook?
- Analyst-style dashboard with live single-post scoring and CSV batch scoring
- TF-IDF word n-grams + Logistic Regression baseline
- NLTK TweetTokenizer preprocessing designed for social text
- Holdout evaluation: accuracy, macro precision/recall/F1, per-class report and confusion matrix
- Dataset composition and post-length analysis
- CSV exports for scored posts and cleaned training data
- Model card, limitations and responsible-use notes

## Project structure

```text
twitter_sentiment_enterprise/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── sample_tweets.csv
```

## Run on Windows (PowerShell / VS Code)

If `pip` is not recognized, use `py -m pip` instead.

```powershell
py --version
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

If PowerShell blocks virtual-environment activation, either run this for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Or skip activation and run:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Use your own dataset

Choose **Upload labeled CSV** in the sidebar. Provide a CSV containing:
- A text column (e.g. `text`, `tweet`, `content`)
- A label column (e.g. `sentiment`, `label`)

Accepted sentiment labels include:
- Negative: `negative`, `neg`, `-1`, `0`
- Neutral: `neutral`, `neu`, `1`
- Positive: `positive`, `pos`, `2`, `4`

The app lets you select the text and label columns. At least 15 valid rows and at least 2 examples per class are required for a basic run; substantially more data is needed for meaningful model evaluation.

For batch scoring, upload a CSV containing text only; labels are not required.

## Model workflow

1. Clean and normalize text (URLs and user mentions are normalized; hashtags and emojis are retained where possible).
2. Tokenize social text using NLTK `TweetTokenizer`.
3. Convert text into TF-IDF unigram and bigram features.
4. Train a class-balanced Logistic Regression classifier.
5. Hold out a stratified test subset and report class-wise metrics.
6. Score new posts and export predictions with confidence values.

## Important evaluation caveat

The default demo dataset is small, hand-written and intended only to demonstrate the application flow. Its score is not evidence of real-world performance. For a portfolio or industry evaluation, use a suitably licensed, representative, labeled dataset. Prevent leakage between train and test data (for example, near-duplicate posts or posts from the same campaign appearing in both splits). If timestamps exist, prefer a chronological split and evaluate on later data.

## Data and API limitations

This project does not connect to the X/Twitter API and does not scrape posts. Import only data you are authorized to use. API access, platform terms, privacy, retention and deletion requirements must be assessed separately before production use.

## Future production upgrades

- Time-based and group-aware validation
- Calibration and abstention for low-confidence predictions
- Multilingual models for Hindi/English code-switching
- Drift monitoring and human review queues
- Experiment tracking, model versioning and reproducible training
- API service, authentication, audit logs and deployment hardening
