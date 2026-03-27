import logging

import requests

from nlp_stylometry.config import get_settings
from nlp_stylometry.preprocessing import clean_text

logger = logging.getLogger(__name__)

settings = get_settings()


class PrepareData:
    def _fetch_titles_per_author(self) -> dict[str, list[str]]:
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

        results: dict[str, list[str]] = {}
        for item in response.json():
            author = item.get("author")
            if author in ["Adam Mickiewicz", "Juliusz Słowacki", "Zygmunt Krasiński"]:
                results.setdefault(author, []).append(item.get("href"))

        logger.info(
            "Found %d authors: %s",
            len(results),
            list(results.keys()),
        )
        for author, titles in results.items():
            logger.debug("Author %r has %d title(s)", author, len(titles))

        return results

    def _get_text(self, format: str, all: bool = False) -> dict[str, list[str]]:
        if format not in ["txt", "pdf", "xml", "html"]:
            raise ValueError(
                "Format must be one of: 'txt', 'pdf', 'xml', 'html'"
            )

        titles_per_author = self._fetch_titles_per_author()
        result: dict[str, list[str]] = {}

        for author, items in titles_per_author.items():
            logger.info(
                "Fetching texts for %r (%d title(s), all=%s)",
                author,
                len(items),
                all,
            )
            result[author] = []

            for item in items:
                logger.debug("Fetching metadata from %s", item)
                try:
                    response = requests.get(item, timeout=30)
                except requests.RequestException as e:
                    logger.error("Request failed for %s: %s", item, e)
                    continue

                if response.status_code != 200:
                    logger.warning(
                        "Failed to fetch metadata from %s. Status code: %d",
                        item,
                        response.status_code,
                    )
                    continue

                data = response.json()
                file_url = data.get(format)

                if not file_url:
                    logger.warning("Format %r not available for %s", format, item)
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
                    result[author].append(clean_text(text_part))
                elif format == "pdf":
                    result[author].append(content_response.content)
                else:
                    result[author].append(content_response.text)

                logger.info("Fetched %s text for %r from %s", format, author, file_url)

                if not all:
                    break

        return result

    def save_data(self, format: str, all: bool = False) -> None:
        logger.info("Starting save_data (format=%r, all=%s)", format, all)
        data = self._get_text(format, all)

        for author, texts in data.items():
            path = settings.data_dir / author.replace(" ", "_")
            path.mkdir(parents=True, exist_ok=True)
            logger.info("Saving %d file(s) for %r to %s", len(texts), author, path)

            for idx, text in enumerate(texts):
                filename = path / f"{idx + 1}.{format}"
                mode = "wb" if format == "pdf" else "w"
                encoding = None if format == "pdf" else "utf-8"

                try:
                    with open(filename, mode, encoding=encoding) as file:
                        file.write(text)
                    logger.debug("Saved %s", filename)
                except OSError as e:
                    logger.error("Failed to write %s: %s", filename, e)
                    raise

        logger.info("save_data complete")
