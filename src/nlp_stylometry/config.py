from pydantic_settings import BaseSettings
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    api_url: str = "https://wolnelektury.pl/api/books/"
    data_dir: Path = BASE_DIR / "data"


def get_settings() -> Settings:
    return Settings()