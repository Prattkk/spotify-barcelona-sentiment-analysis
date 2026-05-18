"""
Spotify × FC Barcelona Sentiment Analysis
Step 2 — RoBERTa Sentiment Classification

Loads the cardiffnlp/twitter-roberta-base-sentiment model from HuggingFace,
runs batch inference over collected Reddit posts, and compares against
a VADER lexicon baseline.

Usage:
    python sentiment_analysis.py

Input:
    data/raw_reddit_posts.csv   (from data_collection.py)

Output:
    data/sentiment_results.csv  (original fields + sentiment label + confidence)
    data/vader_baseline.csv     (VADER scores for accuracy comparison)
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROBERTA_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment"
LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
}

CONFIDENCE_THRESHOLD = 0.65    # predictions below this are flagged 'Low Confidence'
BATCH_SIZE = 32                # tune based on available GPU/CPU memory
MAX_TOKEN_LENGTH = 512         # RoBERTa hard limit

INPUT_PATH = Path("data/raw_reddit_posts.csv")
OUTPUT_PATH = Path("data/sentiment_results.csv")
VADER_PATH = Path("data/vader_baseline.csv")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_roberta_pipeline():
    """
    Load the RoBERTa sentiment classification pipeline from HuggingFace.

    Uses the cardiffnlp/twitter-roberta-base-sentiment checkpoint, which was
    fine-tuned on 58M tweets — a strong domain match for informal social text.

    Downloads model weights on first run (~500MB); cached locally after that.

    Returns:
        transformers.Pipeline: Text classification pipeline ready for inference.

    Raises:
        OSError: If the model cannot be downloaded or loaded from cache.
    """
    # Import deferred — transformers is a heavy dependency
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

    logger.info("Loading RoBERTa model: %s", ROBERTA_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL_ID)

    classifier = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
        top_k=1,               # return only the top predicted label
    )
    logger.info("RoBERTa pipeline loaded successfully")
    return classifier


def load_vader_analyzer():
    """
    Load the VADER sentiment intensity analyzer for baseline comparison.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based
    model tuned for social media text. Used here as a baseline to quantify
    RoBERTa's accuracy improvement on this domain.

    Returns:
        vaderSentiment.SentimentIntensityAnalyzer: Initialized VADER analyzer.

    Raises:
        LookupError: If the VADER lexicon resource is not downloaded.
                     Run: python -m nltk.downloader vader_lexicon
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    logger.info("Loading VADER analyzer")
    return SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------

def prepare_text(title: str, selftext: str) -> str:
    """
    Combine post title and body into a single inference string.

    Concatenates title and selftext with a separator. Truncation to
    MAX_TOKEN_LENGTH is handled by the HuggingFace tokenizer downstream.

    Args:
        title:    Reddit post title.
        selftext: Reddit post body text.

    Returns:
        Combined text string for inference. Returns title alone if selftext
        is empty, deleted, or removed.
    """
    body = selftext.strip()
    if body in ("", "[deleted]", "[removed]"):
        return title.strip()
    return f"{title.strip()} {body}"[:2000]    # pre-truncate to avoid tokenizer overhead


def is_valid_text(text: str) -> bool:
    """
    Return True if text is non-empty and long enough for meaningful inference.

    Filters out single-word posts, pure URLs, and whitespace-only strings
    that would produce unreliable sentiment scores.

    Args:
        text: Prepared text string.

    Returns:
        bool: True if text passes validity checks.
    """
    cleaned = text.strip()
    if not cleaned:
        return False
    word_count = len(cleaned.split())
    return word_count >= 3     # at least 3 words for meaningful inference


# ---------------------------------------------------------------------------
# RoBERTa inference
# ---------------------------------------------------------------------------

def run_roberta_batch(
    texts: list[str],
    classifier,
) -> list[dict[str, str | float]]:
    """
    Run batch RoBERTa inference over a list of text strings.

    Processes texts in batches of BATCH_SIZE. Each result contains:
        - label:      'Positive', 'Neutral', or 'Negative'
        - confidence: float 0.0–1.0 (softmax probability of top label)
        - flagged:    True if confidence < CONFIDENCE_THRESHOLD

    Args:
        texts:      List of prepared text strings for inference.
        classifier: Loaded HuggingFace text-classification pipeline.

    Returns:
        List of result dicts, one per input text, preserving order.
    """
    results: list[dict] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        logger.info("Inferring batch %d–%d / %d", i + 1, i + len(batch), len(texts))

        try:
            raw_preds = classifier(batch)
        except Exception as exc:
            logger.error("Batch inference failed at index %d: %s", i, exc)
            # Append low-confidence neutral placeholders so row count stays aligned
            results.extend(
                {"label": "Neutral", "confidence": 0.0, "flagged": True}
                for _ in batch
            )
            continue

        for pred in raw_preds:
            # Pipeline returns [[{label, score}]] when top_k=1
            top = pred[0] if isinstance(pred, list) else pred
            label = LABEL_MAP.get(top["label"], top["label"])
            confidence = round(top["score"], 4)
            results.append({
                "label": label,
                "confidence": confidence,
                "flagged": confidence < CONFIDENCE_THRESHOLD,
            })

    return results


# ---------------------------------------------------------------------------
# VADER baseline
# ---------------------------------------------------------------------------

def run_vader_baseline(
    texts: list[str],
    analyzer,
) -> list[dict[str, str | float]]:
    """
    Score each text with VADER and map compound score to a sentiment label.

    VADER compound score ranges from -1 (most negative) to +1 (most positive).
    Mapping thresholds (standard VADER convention):
        compound >= 0.05  → Positive
        compound <= -0.05 → Negative
        otherwise         → Neutral

    Args:
        texts:    List of text strings.
        analyzer: Loaded VADER SentimentIntensityAnalyzer.

    Returns:
        List of dicts with keys: label (str), compound (float).
    """
    results = []
    for text in texts:
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        results.append({"vader_label": label, "vader_compound": round(compound, 4)})
    return results


# ---------------------------------------------------------------------------
# Aggregation & export
# ---------------------------------------------------------------------------

def merge_results(
    df: pd.DataFrame,
    roberta_results: list[dict],
    vader_results: list[dict],
) -> pd.DataFrame:
    """
    Merge RoBERTa and VADER predictions back into the original DataFrame.

    Args:
        df:               Original posts DataFrame (aligned row-for-row with results).
        roberta_results:  List of RoBERTa prediction dicts.
        vader_results:    List of VADER prediction dicts.

    Returns:
        DataFrame with appended sentiment columns ready for export.
    """
    roberta_df = pd.DataFrame(roberta_results).rename(
        columns={"label": "roberta_label", "confidence": "roberta_confidence", "flagged": "low_confidence"}
    )
    vader_df = pd.DataFrame(vader_results)

    return pd.concat(
        [df.reset_index(drop=True), roberta_df, vader_df],
        axis=1,
    )


def compute_accuracy_comparison(df: pd.DataFrame) -> None:
    """
    Log a simple agreement rate between RoBERTa and VADER labels.

    Agreement rate is used as a proxy for inter-model consistency.
    Divergence highlights posts where the models disagree — often sarcasm
    or domain-specific language that VADER misclassifies.

    Args:
        df: Merged DataFrame containing 'roberta_label' and 'vader_label' columns.
    """
    agreement = (df["roberta_label"] == df["vader_label"]).mean()
    logger.info("RoBERTa vs VADER agreement rate: %.1f%%", agreement * 100)
    logger.info(
        "RoBERTa distribution:\n%s",
        df["roberta_label"].value_counts(normalize=True).round(3).to_string(),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate full sentiment classification pipeline.

    Loads data → prepares text → runs RoBERTa inference →
    runs VADER baseline → merges results → exports CSV.
    """
    logger.info("Loading input data from %s", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)
    logger.info("Loaded %d posts", len(df))

    # Prepare text and filter invalid rows
    df["text"] = df.apply(
        lambda row: prepare_text(str(row.get("title", "")), str(row.get("selftext", ""))),
        axis=1,
    )
    valid_mask = df["text"].apply(is_valid_text)
    df_valid = df[valid_mask].copy()
    logger.info("Valid posts for inference: %d / %d", len(df_valid), len(df))

    texts = df_valid["text"].tolist()

    # RoBERTa inference
    classifier = load_roberta_pipeline()
    roberta_results = run_roberta_batch(texts, classifier)

    # VADER baseline
    vader_analyzer = load_vader_analyzer()
    vader_results = run_vader_baseline(texts, vader_analyzer)

    # Merge and export
    df_out = merge_results(df_valid, roberta_results, vader_results)
    compute_accuracy_comparison(df_out)

    df_out.to_csv(OUTPUT_PATH, index=False)
    logger.info("Results exported to %s", OUTPUT_PATH)

    # Save VADER-only baseline for separate comparison
    vader_df = df_valid[["post_id", "text"]].copy()
    vader_df = pd.concat([vader_df.reset_index(drop=True), pd.DataFrame(vader_results)], axis=1)
    vader_df.to_csv(VADER_PATH, index=False)
    logger.info("VADER baseline exported to %s", VADER_PATH)


if __name__ == "__main__":
    main()
