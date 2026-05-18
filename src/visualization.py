"""
Spotify × FC Barcelona Sentiment Analysis
Step 3 — Visualization & Insights

Generates publication-quality charts from RoBERTa sentiment results:
    1. Sentiment distribution bar chart
    2. Time-series sentiment trend line chart with event annotations
    3. Subreddit-level sentiment heatmap
    4. Rolling sentiment ratio chart

Usage:
    python visualization.py

Input:
    data/sentiment_results.csv  (from sentiment_analysis.py)

Output:
    figures/sentiment_distribution.png
    figures/sentiment_trend.png
    figures/subreddit_heatmap.png
    figures/rolling_sentiment.png
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_PATH = Path("data/sentiment_results.csv")
FIGURES_DIR = Path("figures")

# Color palette — Spotify green, neutral grey, negative red
SENTIMENT_COLORS = {
    "Positive": "#1DB954",    # Spotify green
    "Neutral":  "#B3B3B3",    # Spotify grey
    "Negative": "#A50044",    # Barcelona red
}

# Key Spotify × Barcelona partnership milestone dates for event annotation
PARTNERSHIP_EVENTS = {
    "2022-07-01": "Naming rights\nannounced",
    "2022-07-15": "New jersey\nrevealed",
    "2023-08-01": "Camp Nou\nrenovation",
    "2024-09-01": "UCL group\nstage",
}

DPI = 150
FIGURE_SIZE = (12, 6)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup() -> pd.DataFrame:
    """
    Load sentiment results and prepare datetime index for time-series plots.

    Converts 'created_utc' Unix timestamps to UTC datetime.
    Sets 'post_date' as a daily-resolution date column for aggregation.

    Returns:
        DataFrame with 'roberta_label', 'created_utc', and 'post_date' columns.

    Raises:
        FileNotFoundError: If sentiment_results.csv does not exist.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({"figure.dpi": DPI, "savefig.bbox": "tight"})

    logger.info("Loading sentiment results from %s", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)

    df["datetime"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    df["post_date"] = df["datetime"].dt.normalize()
    logger.info("Loaded %d labeled posts", len(df))
    return df


# ---------------------------------------------------------------------------
# Chart 1 — Sentiment distribution
# ---------------------------------------------------------------------------

def plot_sentiment_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Render a horizontal bar chart of overall sentiment label distribution.

    Displays raw counts and percentage labels on each bar.
    Color-coded by sentiment class (Positive=green, Neutral=grey, Negative=red).

    Args:
        df:   Sentiment results DataFrame containing 'roberta_label' column.
        save: If True, writes PNG to figures/sentiment_distribution.png.

    Returns:
        Matplotlib Figure object.
    """
    counts = df["roberta_label"].value_counts().reindex(["Positive", "Neutral", "Negative"])
    pcts = (counts / counts.sum() * 100).round(1)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        counts.index,
        counts.values,
        color=[SENTIMENT_COLORS[l] for l in counts.index],
        edgecolor="white",
        height=0.55,
    )

    # Annotate bars with count and percentage
    for bar, pct in zip(bars, pcts):
        width = bar.get_width()
        ax.text(
            width + counts.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width):,}  ({pct}%)",
            va="center", ha="left", fontsize=11, color="#333",
        )

    ax.set_xlabel("Number of Posts", labelpad=10)
    ax.set_title(
        "Spotify × FC Barcelona — Reddit Sentiment Distribution\n"
        "RoBERTa transformer · 10,000+ posts",
        fontsize=13, pad=14,
    )
    ax.set_xlim(0, counts.max() * 1.25)
    ax.invert_yaxis()

    if save:
        path = FIGURES_DIR / "sentiment_distribution.png"
        fig.savefig(path)
        logger.info("Saved %s", path)

    return fig


# ---------------------------------------------------------------------------
# Chart 2 — Time-series sentiment trend
# ---------------------------------------------------------------------------

def plot_sentiment_trend(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Render a multi-line time-series chart of daily sentiment counts.

    One line per sentiment class. X-axis is date; Y-axis is daily post count.
    Partnership milestone events are annotated as vertical dashed lines.

    Args:
        df:   Sentiment results DataFrame with 'post_date' and 'roberta_label'.
        save: If True, writes PNG to figures/sentiment_trend.png.

    Returns:
        Matplotlib Figure object.
    """
    # Aggregate daily counts per sentiment class
    daily = (
        df.groupby(["post_date", "roberta_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for label, color in SENTIMENT_COLORS.items():
        if label in daily.columns:
            ax.plot(
                daily.index,
                daily[label].rolling(7, min_periods=1).mean(),   # 7-day rolling average
                label=f"{label} (7-day avg)",
                color=color,
                linewidth=2,
            )

    # Annotate partnership milestones
    _annotate_events(ax, df["post_date"].min(), df["post_date"].max())

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30, ha="right")

    ax.set_xlabel("Date", labelpad=10)
    ax.set_ylabel("Post Count (7-day avg)", labelpad=10)
    ax.set_title(
        "Spotify × FC Barcelona — Sentiment Trend Over Time",
        fontsize=13, pad=14,
    )
    ax.legend(loc="upper left", framealpha=0.85)

    if save:
        path = FIGURES_DIR / "sentiment_trend.png"
        fig.savefig(path)
        logger.info("Saved %s", path)

    return fig


# ---------------------------------------------------------------------------
# Chart 3 — Subreddit sentiment heatmap
# ---------------------------------------------------------------------------

def plot_subreddit_heatmap(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Render a heatmap of sentiment distribution broken down by subreddit.

    Rows: subreddits (r/soccer, r/fcbarcelona, r/spotify).
    Columns: sentiment classes.
    Cell values: percentage of posts in that subreddit with that label.

    Args:
        df:   Sentiment results DataFrame with 'subreddit' and 'roberta_label'.
        save: If True, writes PNG to figures/subreddit_heatmap.png.

    Returns:
        Matplotlib Figure object.
    """
    pivot = (
        df.groupby(["subreddit", "roberta_label"])
        .size()
        .unstack(fill_value=0)
    )
    # Normalize to row percentages
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        pivot_pct,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        linewidths=0.5,
        cbar_kws={"label": "% of Posts"},
        ax=ax,
        vmin=0,
        vmax=100,
    )
    ax.set_title(
        "Sentiment Distribution by Subreddit (%)",
        fontsize=13, pad=14,
    )
    ax.set_xlabel("Sentiment", labelpad=10)
    ax.set_ylabel("Subreddit", labelpad=10)

    if save:
        path = FIGURES_DIR / "subreddit_heatmap.png"
        fig.savefig(path)
        logger.info("Saved %s", path)

    return fig


# ---------------------------------------------------------------------------
# Chart 4 — Rolling positive sentiment ratio
# ---------------------------------------------------------------------------

def plot_rolling_sentiment_ratio(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Render a rolling positive sentiment ratio (Positive / Total) time-series.

    This is the primary KPI for tracking partnership brand health over time.
    A ratio above 0.5 indicates net-positive fan reception for that period.

    Args:
        df:   Sentiment results DataFrame with 'post_date' and 'roberta_label'.
        save: If True, writes PNG to figures/rolling_sentiment.png.

    Returns:
        Matplotlib Figure object.
    """
    daily = (
        df.groupby(["post_date", "roberta_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
    )

    daily["total"] = daily.sum(axis=1)
    daily["positive_ratio"] = daily["Positive"] / daily["total"].replace(0, pd.NA)

    rolling_ratio = daily["positive_ratio"].rolling(14, min_periods=3).mean()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.fill_between(rolling_ratio.index, rolling_ratio, 0.5,
                    where=(rolling_ratio >= 0.5), alpha=0.25, color="#1DB954", label="Net Positive")
    ax.fill_between(rolling_ratio.index, rolling_ratio, 0.5,
                    where=(rolling_ratio < 0.5), alpha=0.25, color="#A50044", label="Net Negative")
    ax.plot(rolling_ratio.index, rolling_ratio, color="#333333", linewidth=1.5, label="14-day avg ratio")
    ax.axhline(0.5, color="#999", linewidth=1, linestyle="--", label="Neutral threshold (0.5)")

    _annotate_events(ax, daily.index.min(), daily.index.max())

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30, ha="right")

    ax.set_xlabel("Date", labelpad=10)
    ax.set_ylabel("Positive Sentiment Ratio", labelpad=10)
    ax.set_title(
        "Spotify × FC Barcelona — Rolling Positive Sentiment Ratio (14-day)",
        fontsize=13, pad=14,
    )
    ax.legend(loc="lower left", framealpha=0.85)

    if save:
        path = FIGURES_DIR / "rolling_sentiment.png"
        fig.savefig(path)
        logger.info("Saved %s", path)

    return fig


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _annotate_events(
    ax: plt.Axes,
    date_min: pd.Timestamp,
    date_max: pd.Timestamp,
) -> None:
    """
    Draw vertical dashed lines for Spotify × Barcelona milestones.

    Skips events outside the data's date range to avoid rendering
    annotations beyond the visible chart area.

    Args:
        ax:       Matplotlib Axes to annotate.
        date_min: Earliest date in the dataset.
        date_max: Latest date in the dataset.
    """
    for date_str, label in PARTNERSHIP_EVENTS.items():
        event_date = pd.Timestamp(date_str, tz="UTC")
        if not (date_min <= event_date <= date_max):
            continue
        ax.axvline(event_date, color="#666", linewidth=1, linestyle=":", alpha=0.7)
        ax.text(
            event_date, ax.get_ylim()[1] * 0.97,
            label,
            ha="center", va="top", fontsize=8, color="#555",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Generate all four sentiment visualization charts from the results CSV.
    """
    df = setup()

    plot_sentiment_distribution(df)
    plot_sentiment_trend(df)
    plot_subreddit_heatmap(df)
    plot_rolling_sentiment_ratio(df)

    logger.info("All figures saved to %s/", FIGURES_DIR)


if __name__ == "__main__":
    main()
