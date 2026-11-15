"""Build and query a regional Indian OTT content catalog."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
RAW_CATALOG = ROOT / "data" / "raw" / "Indian_Web_Series_2014_2025_Episodes_Clean.csv"
CATALOG_DB = ROOT / "data" / "processed" / "ott_catalog.duckdb"
MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL: SentenceTransformer | None = None

PLATFORM_ALIASES = {
    "hotstar": "JioHotstar",
    "jio cinema": "JioHotstar",
    "jiocinema": "JioHotstar",
    "jio hotstar": "JioHotstar",
    "jiohotstar": "JioHotstar",
    "sony liv": "SonyLIV",
    "sonyliv": "SonyLIV",
    "prime video": "Prime Video",
    "amazon prime video": "Prime Video",
    "zee5": "ZEE5",
}
LANGUAGE_HINTS = {
    "malayalam": "Malayalam",
    "telugu": "Telugu",
    "tamil": "Tamil",
    "kannada": "Kannada",
    "bengali": "Bengali",
    "marathi": "Marathi",
    "punjabi": "Punjabi",
    "gujarati": "Gujarati",
    "hindi": "Hindi",
}


def _text(value: Any, fallback: str = "Unknown") -> str:
    if value is None:
        return fallback
    result = str(value).strip()
    return result if result and result.lower() not in {"null", "nan", "none"} else fallback


def _normalise_platform(value: Any) -> str:
    platform = _text(value, "Unknown / Indie OTT")
    return PLATFORM_ALIASES.get(platform.lower(), platform)


def _infer_language(title: str, genres: str, platform: str) -> str:
    haystack = f"{title} {genres} {platform}".lower()
    for hint, language in LANGUAGE_HINTS.items():
        if re.search(rf"\b{re.escape(hint)}\b", haystack):
            return language
    return "Hindi"


def _column(frame: pl.DataFrame, name: str, fallback: str) -> pl.Series:
    if name in frame.columns:
        return frame.get_column(name).cast(pl.String, strict=False).fill_null(fallback)
    return pl.Series(name, [fallback] * frame.height)


def aggregate_catalog(raw_path: Path = RAW_CATALOG) -> pl.DataFrame:
    """Read episode rows and return one enriched record per series."""
    schema = pl.scan_csv(raw_path).collect_schema()
    frame = pl.read_csv(raw_path, infer_schema_length=2_000)
    title = _column(frame, "series_title", "Untitled Indian Series").str.strip_chars()
    title = title.replace("", "Untitled Indian Series")
    series_id = _column(frame, "series_id", "").str.strip_chars()
    series_id = series_id.zip_with(series_id != "", title)
    platform = _column(frame, "network_service", "Unknown / Indie OTT").map_elements(
        _normalise_platform, return_dtype=pl.String
    )
    genre = _column(frame, "primary_genre", "Drama")
    all_genres = _column(frame, "all_genres", "")
    genre_text = pl.when(all_genres != "").then(all_genres).otherwise(genre)
    language = pl.struct([title.alias("title"), genre_text.alias("genres"), platform.alias("platform")]).map_elements(
        lambda row: _infer_language(row["title"], row["genres"], row["platform"]),
        return_dtype=pl.String,
    )
    prepared = frame.with_columns(
        title.alias("series_title"),
        series_id.alias("_series_key"),
        platform.alias("platform"),
        genre.alias("genre"),
        genre_text.alias("_genre_text"),
        language.alias("language"),
    )

    episode_id = "episode_id" if "episode_id" in schema.names() else None
    rating = (
        pl.col("imdb_rating").cast(pl.Float64, strict=False).mean().alias("avg_rating")
        if "imdb_rating" in schema.names()
        else pl.lit(None, dtype=pl.Float64).alias("avg_rating")
    )
    count_expr = (
        pl.col(episode_id).n_unique().alias("episode_count")
        if episode_id
        else pl.len().alias("episode_count")
    )
    grouped = (
        prepared.group_by("_series_key", "series_title", "platform", "language", "genre")
        .agg(count_expr, rating)
        .with_columns(pl.col("avg_rating").round(2))
    )
    return grouped.with_columns(
        pl.format(
            "{} is a {} {} series on {}. It has {} episodes with an average IMDb rating of {}. "
            "The catalog genres are {}. This regional recommendation is suitable for {} viewers.",
            pl.col("series_title"),
            pl.col("language"),
            pl.col("genre"),
            pl.col("platform"),
            pl.col("episode_count"),
            pl.col("avg_rating").fill_null("not available"),
            pl.col("genre"),
            pl.col("language"),
        ).alias("searchable_text")
    ).select(
        "series_title",
        "platform",
        "language",
        "genre",
        "episode_count",
        "avg_rating",
        "searchable_text",
    )


def _model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def build_catalog(
    raw_path: Path = RAW_CATALOG, database_path: Path = CATALOG_DB
) -> Path:
    """Aggregate, embed, and persist the catalog in DuckDB."""
    catalog = aggregate_catalog(raw_path)
    embeddings = _model().encode(
        catalog["searchable_text"].to_list(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database_path)) as connection:
        connection.execute("DROP TABLE IF EXISTS ott_catalog")
        connection.execute(
            """
            CREATE TABLE ott_catalog (
                series_title VARCHAR,
                platform VARCHAR,
                language VARCHAR,
                genre VARCHAR,
                episode_count INTEGER,
                avg_rating DOUBLE,
                searchable_text VARCHAR,
                embedding FLOAT[]
            )
            """
        )
        rows = [
            (*row, embedding.tolist())
            for row, embedding in zip(catalog.iter_rows(), embeddings)
        ]
        connection.executemany(
            "INSERT INTO ott_catalog VALUES (?, ?, ?, ?, ?, ?, ?, ?::FLOAT[])",
            rows,
        )
    return database_path


def query_indian_ott_catalog(
    preferred_language: str,
    mood_query: str,
    exclude_titles: list[str] | None = None,
    top_k: int = 3,
) -> list[dict]:
    """Return regional semantic matches, boosting the requested language."""
    if top_k < 1:
        return []
    if not CATALOG_DB.exists():
        raise FileNotFoundError(f"Catalog database does not exist: {CATALOG_DB}")
    query_embedding = _model().encode(
        f"{preferred_language} {mood_query}",
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    excluded = {title.casefold() for title in (exclude_titles or [])}
    with duckdb.connect(str(CATALOG_DB), read_only=True) as connection:
        rows = connection.execute(
            "SELECT series_title, platform, language, genre, episode_count, avg_rating, searchable_text, embedding "
            "FROM ott_catalog"
        ).fetchall()
    results: list[dict] = []
    requested = preferred_language.strip().casefold()
    for row in rows:
        if row[0].casefold() in excluded:
            continue
        embedding = np.asarray(row[7], dtype=np.float32)
        similarity = float(np.dot(query_embedding, embedding))
        language_boost = 0.18 if row[2].casefold() == requested else 0.0
        results.append(
            {
                "series_title": row[0],
                "platform": row[1],
                "language": row[2],
                "genre": row[3],
                "episode_count": row[4],
                "avg_rating": row[5],
                "searchable_text": row[6],
                "similarity_score": round(similarity + language_boost, 4),
            }
        )
    return sorted(results, key=lambda result: result["similarity_score"], reverse=True)[:top_k]


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="high-adrenaline thriller or crime series after cricket")
    parser.add_argument("--language", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    build_catalog()
    personas_path = ROOT / "data" / "processed" / "flagship_personas.json"
    personas = json.loads(personas_path.read_text(encoding="utf-8"))
    ipl_persona = next(
        persona for persona in personas if persona["primary_churn_driver"] == "IPL_Post_Tournament_Cliff"
    )
    language = args.language or ipl_persona["preferred_language"]
    recommendations = query_indian_ott_catalog(language, args.query, top_k=args.top_k)
    print(json.dumps(recommendations, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    _main()
