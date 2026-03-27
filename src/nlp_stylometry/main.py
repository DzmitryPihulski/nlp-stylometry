import logging

from nlp_stylometry.api.fetch_data import PrepareData


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    prepare_data = PrepareData()
    prepare_data.save_data(format="txt", all=True)


if __name__ == "__main__":
    main()