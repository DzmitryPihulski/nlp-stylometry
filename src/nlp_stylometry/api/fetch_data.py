import json
import logging
from typing import Any

import requests

from nlp_stylometry.config import get_settings
from nlp_stylometry.preprocessing import clean_text

logger = logging.getLogger(__name__)

settings = get_settings()


def _sanitize_dirname(s: str) -> str:
    """Make a string safe for use as a directory name."""
    return s.replace(", ", "_").replace(" ", "_")


class PrepareData:
    def _fetch_titles_per_author(self) -> dict[str, list[dict]]:
        logger.info("Fetching book list from %s", settings.api_url)
        try:
            response = requests.get(settings.api_url, timeout=30)
        except requests.RequestException as e:
            logger.error("Request failed while fetching book list: %s", e)
            raise

        if response.status_code != 200:
            logger.error(
                "Failed to fetch book list from %s. Status code: %d",
                settings.api_url,
                response.status_code,
            )
            raise Exception(
                f"Failed to fetch data from {settings.api_url}. Status code: {response.status_code}"
            )

        results: dict[str, list[dict]] = {}
        all_kinds: set[str] = set()
        all_genres: set[str] = set()

        for item in response.json():
            author = item.get("author")
            if author in ["Adam Mickiewicz", "Juliusz Słowacki", "Zygmunt Krasiński"]:
                kind = item.get("kind", "unknown")
                genre = item.get("genre", "unknown")
                all_kinds.add(kind)
                all_genres.add(genre)
                results.setdefault(author, []).append(
                    {"href": item.get("href"), "kind": kind, "genre": genre}
                )

        logger.info("Unique kinds: %s", all_kinds)
        logger.info("Unique genres: %s", all_genres)
        logger.info("Found %d authors: %s", len(results), list(results.keys()))
        for author, titles in results.items():
            logger.debug("Author %r has %d title(s)", author, len(titles))

        return results

    def _get_text(self, format: str, all: bool = False) -> list[dict[str, Any]]:
        if format not in ["txt", "pdf", "xml", "html"]:
            raise ValueError("Format must be one of: 'txt', 'pdf', 'xml', 'html'")

        titles_per_author = self._fetch_titles_per_author()
        records: list[dict[str, Any]] = []

        for author, items in titles_per_author.items():
            logger.info(
                "Fetching texts for %r (%d title(s), all=%s)",
                author,
                len(items),
                all,
            )
            fetched = 0

            for item in items:
                href = item["href"]
                kind = item["kind"]
                genre = item["genre"]

                logger.debug("Fetching metadata from %s", href)
                try:
                    response = requests.get(href, timeout=30)
                except requests.RequestException as e:
                    logger.error("Request failed for %s: %s", href, e)
                    continue

                if response.status_code != 200:
                    logger.warning(
                        "Failed to fetch metadata from %s. Status code: %d",
                        href,
                        response.status_code,
                    )
                    continue

                data = response.json()
                file_url = data.get(format)
                title = data.get("title", "unknown")

                if not file_url:
                    logger.warning("Format %r not available for %s", format, href)
                    continue

                logger.debug("Fetching %s content from %s", format, file_url)
                try:
                    content_response = requests.get(file_url, timeout=60)
                except requests.RequestException as e:
                    logger.error("Request failed for content %s: %s", file_url, e)
                    continue

                if content_response.status_code != 200:
                    logger.warning(
                        "Failed to fetch content from %s. Status code: %d",
                        file_url,
                        content_response.status_code,
                    )
                    continue

                if format == "txt":
                    text_part = content_response.text.split("-----")[0]
                    content = clean_text(text_part)
                elif format == "pdf":
                    content = content_response.content
                else:
                    content = content_response.text

                records.append(
                    {"author": author, "kind": kind, "genre": genre, "title": title, "content": content}
                )
                fetched += 1
                logger.info(
                    "Fetched %s text for %r from %s (kind=%r, genre=%r)",
                    format, author, file_url, kind, genre,
                )

                if not all:
                    break

            logger.info("Fetched %d text(s) for %r", fetched, author)

        return records

    def save_data(self, format: str, all: bool = False) -> None:
        logger.info("Starting save_data (format=%r, all=%s)", format, all)
        records = self._get_text(format, all)

        # Track per-(author, kind, genre) index for sequential filenames
        counters: dict[tuple[str, str, str], int] = {}

        for record in records:
            author = record["author"]
            kind = _sanitize_dirname(record["kind"])
            genre = _sanitize_dirname(record["genre"])

            path = (
                settings.data_dir
                / author.replace(" ", "_")
                / kind
                / genre
            )
            path.mkdir(parents=True, exist_ok=True)

            key = (author, kind, genre)
            counters[key] = counters.get(key, 0) + 1
            idx = counters[key]
            filename = path / f"{idx}.{format}"
            sidecar = path / f"{idx}.json"

            mode = "wb" if format == "pdf" else "w"
            encoding = None if format == "pdf" else "utf-8"

            try:
                with open(filename, mode, encoding=encoding) as file:
                    file.write(record["content"])
                sidecar.write_text(
                    json.dumps({"title": record["title"]}, ensure_ascii=False),
                    encoding="utf-8",
                )
                logger.debug("Saved %s (title=%r)", filename, record["title"])
            except OSError as e:
                logger.error("Failed to write %s: %s", filename, e)
                raise

        logger.info("save_data complete (%d file(s) saved)", len(records))
