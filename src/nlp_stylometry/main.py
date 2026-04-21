import json
import logging

from nlp_stylometry.api.fetch_data import PrepareData
from nlp_stylometry.config import get_settings
from nlp_stylometry.preprocess import build_corpus

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    settings = get_settings()

    # Step 1: download raw texts from wolnelektury.pl
    logger.info("Step 1/2: fetching texts")
    PrepareData().save_data(format="txt", all=True)

    # Step 2: preprocess and balance into model-ready chunks
    logger.info("Step 2/2: preprocessing")
    corpora = build_corpus(settings.data_dir)

    dataset = [
        {"author": corpus.author, "kind": corpus.kind, "genre": corpus.genre, "title": corpus.title, "text": chunk}
        for corpus in corpora
        for chunk in corpus.chunks
    ]

    out_path = settings.data_dir / "dataset.json"
    out_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Saved %d samples (%d authors) to %s",
        len(dataset),
        len(corpora),
        out_path,
    )


if __name__ == "__main__":
    main()