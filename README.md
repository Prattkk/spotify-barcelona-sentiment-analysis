# 🎵 Spotify × FC Barcelona — Social Sentiment & Partnership Analytics

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-FF6B6B?style=for-the-badge)
![RoBERTa](https://img.shields.io/badge/RoBERTa-7B2FBE?style=for-the-badge)
![Reddit API](https://img.shields.io/badge/Reddit_API-FF4500?style=for-the-badge&logo=reddit&logoColor=white)
![Marketing Analytics](https://img.shields.io/badge/Marketing_Analytics-0052CC?style=for-the-badge)

## 📌 Overview
Applied a RoBERTa transformer model (cardiffnlp/twitter-roberta-base-sentiment,
trained on 58M tweets) to 289 Reddit comments from 226 unique authors across
r/Barca, r/soccer, and r/football to measure fan sentiment around the
Spotify × FC Barcelona partnership. Combined with event study methodology
using CRSP stock data, Spotify quarterly report analysis, and Google Trends
to deliver a comprehensive data-driven assessment of the partnership's impact
on fans, investors, and global brand perception.

**Published findings:** [Medium Article](https://medium.com/@vermaprateek1109/when-spotify-bought-a-football-clubs-soul-was-it-worth-it-31272e4820a4)

Co-authors: Prateek Verma, Abdul Ali, Juan Zapata,
Samyak Pokharna & Ammratansh Ghildyal | Streaming XI
Gies College of Business, University of Illinois

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

### Sentiment Analysis
| Event | Net Sentiment | Positive | Negative |
|-------|--------------|----------|----------|
| Deal Announcement | +1.4% | ~20% | ~17% |
| Stadium Renaming | −70.8% | ~8% | ~37% |

### By Subreddit
| Subreddit | Positive | Negative | Avg Upvotes |
|-----------|----------|----------|-------------|
| r/soccer | 20% | 20% | 3.0 |
| r/football | 8% | 32% | — |
| r/Barca | Highest engagement | — | 13.8 |

### Stock Market Reaction
- Cumulative Abnormal Return vs Value-Weighted Index: **+2.21%**
- Cumulative Abnormal Return vs S&P 500: **+4.15%**
- Result: Market did not punish the announcement

### User Growth (Pre vs Post Deal)
| Metric | Before Deal | After Deal |
|--------|-------------|------------|
| MAUs | 356M → growing | 602M → growing |
| Quarterly MAU Growth | 4.36% | 5.27% |
| Premium Subscriber Adds/Quarter | 6M | 7.71M |

### Google Trends — Top Countries Post-Announcement
| Rank | Country | Score |
|------|---------|-------|
| 1 | Spain | 100 |
| 2 | Sweden | 50 |
| 3 | Netherlands | — |
| 4 | Germany | — |
| 5 | USA | 5 |

## 🛠️ Tech Stack
- **Sentiment Model:** RoBERTa (cardiffnlp/twitter-roberta-base-sentiment)
  trained on 58 million tweets
- **Baseline Model:** VADER lexicon-based sentiment scoring
- **Data Collection:** Arctic Shift API (Reddit), CRSP/WRDS,
  Spotify 10-Q reports, Google Trends
- **Stock Analysis:** Market model event study (value-weighted index + S&P 500)
- **Language:** Python
- **Libraries:** Transformers (HuggingFace), Pandas, NumPy,
  NLTK, Matplotlib, Seaborn, langdetect
- **Confidence Threshold:** 60% (retained 76.8% of comments)
- **Course:** BADM 534 — Marketing Analytics in Practice,
  Gies College of Business, UIUC

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

## 📝 Full Analysis on Medium
Read the complete data story with all findings, charts, and methodology:

**[When Spotify Bought a Football Club's Soul — Was It Worth It?](https://medium.com/@vermaprateek1109/when-spotify-bought-a-football-clubs-soul-was-it-worth-it-31272e4820a4)**

> A data-driven look at what the Spotify × FC Barcelona deal actually
> did for fans, investors, and the world — using RoBERTa sentiment analysis,
> event study methodology, CRSP stock data, and Google Trends.

Key findings published in the article:
- 📉 Stadium renaming net sentiment: −70.8% (fans rejected the Camp Nou rename)
- 📈 Deal announcement net sentiment: +1.4% (fans tolerated the partnership)
- 📊 Spotify cumulative abnormal return: +4.15% vs S&P 500 post-announcement
- 🌍 Spain ranked #1 globally for search interest, US scored just 5/100
- 👥 289 Reddit comments analyzed across r/Barca, r/soccer, r/football

## 📬 Contact
**Prateek Verma** · [LinkedIn](https://linkedin.com/in/prateek-verma-158b35217) · vermaprateek1109@gmail.com
