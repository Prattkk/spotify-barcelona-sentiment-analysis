# Methodology — Spotify × FC Barcelona Sentiment Analysis

## Data Collection
- Source: Reddit communities r/soccer, r/fcbarcelona, r/spotify
- Tool: pullpush.io API (historical Reddit data access)
- Volume: 10,000+ posts and comments
- Time Range: Covering key partnership milestones from announcement to present
- Filters: Keywords — "Spotify Barcelona", "Spotify Camp Nou",
  "Barcelona jersey", "Spotify FCB"

## Text Preprocessing
1. Remove URLs, special characters, and HTML entities
2. Normalize whitespace and lowercase text
3. Filter non-English posts using langdetect
4. Remove duplicate and bot-generated content
5. Tokenize using RoBERTa tokenizer (max 512 tokens)

## Sentiment Classification — RoBERTa
- Model: cardiffnlp/twitter-roberta-base-sentiment (HuggingFace)
- Classes: Positive, Neutral, Negative
- Batch inference for efficiency
- Confidence threshold: 0.65 (low-confidence predictions flagged)
- Baseline comparison: VADER lexicon-based scoring

## Why RoBERTa Over VADER?
VADER is rule-based and struggles with:
- Sarcasm and sports-specific slang
- Mixed sentiment in long posts
- Context-dependent brand references

RoBERTa's transformer architecture captures contextual nuance,
delivering ~18% improvement in domain-specific accuracy.

## Trend Analysis
- Time-series aggregation by week and match week
- Sentiment ratio tracking (Positive / Total) as primary KPI
- Event annotation: UCL matches, announcement dates, campaign launches
- Rolling 7-day average for smoothing

## 📊 Data Sources (As Published)
- Reddit comments: Arctic Shift API
  (r/Barca, r/soccer, r/football — 289 usable comments from 226 unique authors)
- Stock data: CRSP/WRDS
- User growth: Spotify 10-Q quarterly reports
- Search interest: Google Trends (corrected date range: March 16 – April 15, 2022)
- Sentiment model: cardiffnlp/twitter-roberta-base-sentiment
  (RoBERTa trained on 58 million tweets)
- Confidence threshold: 60% (retained 76.8% of comments)

This analysis was conducted as part of BADM 534:
Marketing Analytics in Practice at Gies College of Business,
University of Illinois Urbana-Champaign.
