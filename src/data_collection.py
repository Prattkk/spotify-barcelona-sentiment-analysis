"""
Spotify × FC Barcelona Sentiment Analysis
Step 1 — Data Collection via pullpush.io API

Collects Reddit posts and comments from r/soccer, r/fcbarcelona, and r/spotify
matching Spotify × Barcelona partnership keywords.
Handles pagination, deduplication, and CSV export.

Usage:
    python data_collection.py

Output:
    data/raw_reddit_posts.csv
    data/raw_reddit_comments.csv
"""

import csv
import logging
import time
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Iterator

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://api.pullpush.io/reddit/search"

SUBREDDITS = ["soccer", "fcbarcelona", "spotify"]

KEYWORDS = [
    "Spotify Barcelona",
    "Spotify Camp Nou",
    "Barcelona jersey Spotify",
    "Spotify FCB",
]

OUTPUT_DIR = Path("data")
POSTS_CSV = OUTPUT_DIR / "raw_reddit_posts.csv"
COMMENTS_CSV = OUTPUT_DIR / "raw_reddit_comments.csv"

REQUEST_DELAY_SEC = 1.0        # polite delay between API calls
MAX_RESULTS_PER_PAGE = 100     # pullpush.io max per request
CONFIDENCE_THRESHOLD = 0.65    # used downstream in sentiment step

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RedditPost:
    """Represents a single Reddit submission (post)."""
    post_id: str
    subreddit: str
    title: str
    selftext: str
    author: str
    score: int
    num_comments: int
    created_utc: int         # Unix timestamp
    url: str
    keyword_matched: str     # which search keyword returned this post


@dataclass
class RedditComment:
    """Represents a single Reddit comment."""
    comment_id: str
    post_id: str             # parent submission ID
    subreddit: str
    body: str
    author: str
    score: int
    created_utc: int
    keyword_matched: str


# ---------------------------------------------------------------------------
# API interaction
# ---------------------------------------------------------------------------

def fetch_posts(
    keyword: str,
    subreddit: str,
    after: int | None = None,
    before: int | None = None,
    size: int = MAX_RESULTS_PER_PAGE,
) -> list[dict]:
    """
    Query the pullpush.io submissions endpoint for a keyword + subreddit.

    pullpush.io is a community-maintained alternative to PushShift for
    historical Reddit data. It supports 'q' (full-text search), 'subreddit',
    'after' / 'before' Unix timestamps for pagination.

    Args:
        keyword:    Full-text search term.
        subreddit:  Target subreddit name (without r/ prefix).
        after:      Return posts created after this Unix timestamp.
        before:     Return posts created before this Unix timestamp.
        size:       Number of results per page (max 100).

    Returns:
        List of raw post dicts from the API response.

    Raises:
        requests.HTTPError: On non-2xx API responses.
        requests.Timeout:   If the request exceeds 30 seconds.
    """
    params: dict = {
        "q": keyword,
        "subreddit": subreddit,
        "size": size,
        "sort": "created_utc",
        "sort_type": "asc",
    }
    if after:
        params["after"] = after
    if before:
        params["before"] = before

    response = requests.get(
        f"{BASE_URL}/submission",
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("data", [])


def fetch_comments(
    keyword: str,
    subreddit: str,
    after: int | None = None,
    size: int = MAX_RESULTS_PER_PAGE,
) -> list[dict]:
    """
    Query the pullpush.io comments endpoint for a keyword + subreddit.

    Args:
        keyword:   Full-text search term.
        subreddit: Target subreddit name.
        after:     Pagination cursor (Unix timestamp of last seen comment).
        size:      Results per page.

    Returns:
        List of raw comment dicts from the API response.

    Raises:
        requests.HTTPError: On non-2xx API responses.
        requests.Timeout:   If the request exceeds 30 seconds.
    """
    params: dict = {
        "q": keyword,
        "subreddit": subreddit,
        "size": size,
        "sort": "created_utc",
        "sort_type": "asc",
    }
    if after:
        params["after"] = after

    response = requests.get(
        f"{BASE_URL}/comment",
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("data", [])


def paginate_posts(
    keyword: str,
    subreddit: str,
    max_pages: int = 50,
) -> Iterator[RedditPost]:
    """
    Yield RedditPost objects across all pages for a keyword + subreddit.

    Uses the created_utc of the last result on each page as the 'after'
    cursor for the next request — pullpush.io's standard pagination pattern.

    Args:
        keyword:   Search keyword.
        subreddit: Target subreddit.
        max_pages: Safety cap on number of API pages to fetch.

    Yields:
        RedditPost dataclass instances.
    """
    after: int | None = None
    pages_fetched = 0

    while pages_fetched < max_pages:
        try:
            raw_posts = fetch_posts(keyword, subreddit, after=after)
        except requests.RequestException as exc:
            logger.error("API error on page %d: %s", pages_fetched + 1, exc)
            break

        if not raw_posts:
            logger.info("No more posts for keyword='%s' subreddit='%s'", keyword, subreddit)
            break

        for raw in raw_posts:
            yield _parse_post(raw, keyword)

        # Advance cursor to the timestamp of the last returned post
        after = raw_posts[-1].get("created_utc")
        pages_fetched += 1
        logger.info("Fetched page %d (%d posts)", pages_fetched, len(raw_posts))
        time.sleep(REQUEST_DELAY_SEC)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_post(raw: dict, keyword: str) -> RedditPost:
    """
    Parse a raw pullpush.io API dict into a typed RedditPost.

    Args:
        raw:     Raw dict from API response.
        keyword: Keyword that returned this post (for downstream tracking).

    Returns:
        RedditPost instance.
    """
    return RedditPost(
        post_id=raw.get("id", ""),
        subreddit=raw.get("subreddit", ""),
        title=raw.get("title", ""),
        selftext=raw.get("selftext", ""),
        author=raw.get("author", ""),
        score=int(raw.get("score", 0)),
        num_comments=int(raw.get("num_comments", 0)),
        created_utc=int(raw.get("created_utc", 0)),
        url=raw.get("url", ""),
        keyword_matched=keyword,
    )


def _parse_comment(raw: dict, keyword: str) -> RedditComment:
    """
    Parse a raw pullpush.io API dict into a typed RedditComment.

    Args:
        raw:     Raw dict from API response.
        keyword: Keyword that returned this comment.

    Returns:
        RedditComment instance.
    """
    return RedditComment(
        comment_id=raw.get("id", ""),
        post_id=raw.get("link_id", "").replace("t3_", ""),
        subreddit=raw.get("subreddit", ""),
        body=raw.get("body", ""),
        author=raw.get("author", ""),
        score=int(raw.get("score", 0)),
        created_utc=int(raw.get("created_utc", 0)),
        keyword_matched=keyword,
    )


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_posts(posts: list[RedditPost]) -> list[RedditPost]:
    """
    Remove duplicate posts by post_id (can appear across keyword queries).

    Keeps the first occurrence — order is preserved.

    Args:
        posts: List of RedditPost instances (may contain duplicates).

    Returns:
        Deduplicated list.
    """
    seen: set[str] = set()
    unique: list[RedditPost] = []
    for post in posts:
        if post.post_id not in seen:
            seen.add(post.post_id)
            unique.append(post)
    logger.info("Deduplication: %d → %d posts", len(posts), len(unique))
    return unique


def filter_deleted(posts: list[RedditPost]) -> list[RedditPost]:
    """
    Remove posts where content was deleted or removed by moderators.

    Deleted posts have selftext '[deleted]' or '[removed]'.

    Args:
        posts: List of RedditPost instances.

    Returns:
        Filtered list with deleted/removed posts excluded.
    """
    cleaned = [
        p for p in posts
        if p.selftext not in ("[deleted]", "[removed]", "")
        or p.title  # keep if title is non-empty even if selftext is gone
    ]
    logger.info("Filter deleted: %d → %d posts", len(posts), len(cleaned))
    return cleaned


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_to_csv(records: list[RedditPost | RedditComment], output_path: Path) -> None:
    """
    Write a list of dataclass records to a CSV file.

    Column headers are derived from dataclass field names.
    Creates parent directories if they don't exist.

    Args:
        records:     List of RedditPost or RedditComment instances.
        output_path: Destination file path.
    """
    if not records:
        logger.warning("No records to export to %s", output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [f.name for f in fields(records[0])]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)

    logger.info("Exported %d records to %s", len(records), output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate full data collection across all keywords and subreddits.

    Collects posts and comments, deduplicates, filters deleted content,
    and exports to CSV for use in the sentiment analysis step.
    """
    all_posts: list[RedditPost] = []

    for keyword in KEYWORDS:
        for subreddit in SUBREDDITS:
            logger.info("Collecting posts — keyword='%s' subreddit='%s'", keyword, subreddit)
            posts = list(paginate_posts(keyword, subreddit))
            all_posts.extend(posts)
            logger.info("Collected %d posts so far", len(all_posts))

    all_posts = deduplicate_posts(all_posts)
    all_posts = filter_deleted(all_posts)

    export_to_csv(all_posts, POSTS_CSV)
    logger.info("Data collection complete. Total posts: %d", len(all_posts))


if __name__ == "__main__":
    main()
