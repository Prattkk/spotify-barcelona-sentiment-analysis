# 🎵 Spotify × FC Barcelona — Social Sentiment & Partnership Analytics

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-FF6B6B?style=for-the-badge)
![RoBERTa](https://img.shields.io/badge/RoBERTa-7B2FBE?style=for-the-badge)
![Reddit API](https://img.shields.io/badge/Reddit_API-FF4500?style=for-the-badge&logo=reddit&logoColor=white)
![Marketing Analytics](https://img.shields.io/badge/Marketing_Analytics-0052CC?style=for-the-badge)

## 📌 Overview
Applied a RoBERTa transformer model to Reddit data to measure fan sentiment
around the Spotify × FC Barcelona partnership. Analyzed 10,000+ posts and
comments via the pullpush.io API to surface brand perception insights,
sentiment trends, and marketing effectiveness signals across time.

## ❗ Business Problem
Brand partnerships worth hundreds of millions of dollars are often evaluated
using traditional surveys and media impressions. This project asks:
**what does organic fan sentiment on social media actually tell us
about partnership effectiveness?**

## 💡 Project Highlights
- Collected and processed 10,000+ Reddit posts and comments using pullpush.io API
- Applied RoBERTa transformer model for nuanced 3-class sentiment classification (Positive / Neutral / Negative)
- Tracked sentiment trends across key partnership milestones
- Surfaced actionable brand perception insights for both Spotify and FC Barcelona
- Visualized sentiment distribution, time-series trends, and topic clustering

## 🏗️ Pipeline Architecture

| Phase | Step | Tool |
|-------|------|------|
| 1 | Data Collection | pullpush.io API, Reddit |
| 2 | Text Preprocessing | Python, NLTK, Regex |
| 3 | Sentiment Classification | RoBERTa (HuggingFace Transformers) |
| 4 | Trend Analysis | Pandas, Time-Series Analysis |
| 5 | Visualization | Matplotlib, Seaborn |

See `/architecture/pipeline-diagram.svg` for the full pipeline diagram.

## 📊 Key Findings
- Sentiment distribution across 10,000+ posts revealed dominant fan reactions to jersey branding, stadium naming rights, and digital activations
- Identified peak positive sentiment windows aligned with UCL match weeks
- Negative sentiment spikes correlated with commercial-heavy announcements
- RoBERTa outperformed VADER baseline by ~18% on domain-specific accuracy

## 🛠️ Tech Stack
- **Language:** Python
- **NLP Model:** RoBERTa (HuggingFace Transformers)
- **Data Source:** Reddit via pullpush.io API
- **Libraries:** Transformers, Pandas, NumPy, NLTK, Matplotlib, Seaborn
- **Analysis:** Sentiment Classification, Time-Series, Topic Analysis

## 📁 Repository Structure
```
spotify-barcelona-sentiment-analysis/
├── architecture/
│   └── pipeline-diagram.svg
├── docs/
│   ├── methodology.md
│   └── insights.md
├── src/
│   ├── data_collection.py
│   ├── sentiment_analysis.py
│   └── visualization.py
├── .gitignore
└── LICENSE
```

## 🎯 Business Applications
- **Brand managers** — measure real-time partnership sentiment without surveys
- **Marketing analysts** — identify optimal timing for campaign announcements
- **Sports partnerships** — benchmark fan reception across social platforms
- **Agencies** — deliver data-driven partnership effectiveness reports

## 📬 Contact
**Prateek Verma** · [LinkedIn](https://linkedin.com/in/prateek-verma-158b35217) · vermaprateek1109@gmail.com
