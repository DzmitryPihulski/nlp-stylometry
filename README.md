# nlp-stylometry

Analiza stylistyczna tekstów polskich romantyków (Mickiewicz, Słowacki, Krasiński) na podstawie tekstów z [wolnelektury.pl](https://wolnelektury.pl).

## Wymagania

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Instalacja

```bash
uv sync
```

## Uruchomienie

```bash
uv run nlp-stylometry
```

Pipeline wykonuje dwa kroki:

1. **Pobieranie** — ściąga teksty z API wolnelektury.pl i zapisuje je w `data/{Autor}/`
2. **Przetwarzanie** — filtruje pliki niebędące językiem polskim, usuwa puste pliki, usuwa etykiety postaci i didaskalia z dramatów, dzieli teksty na zdania i tworzy równoważne batche po 100 zdań dla każdego autora.

## Wynik

Plik `data/dataset.json` — lista rekordów gotowych do użycia w modelach:

```json
[
  { "author": "Adam_Mickiewicz", "text": "..." },
  { "author": "Juliusz_Słowacki", "text": "..." },
  ...
]
```

Każdy rekord to 100 zdań od jednego autora. Liczba rekordów jest zbalansowana — każdy autor ma tyle samo próbek.

## Wczytanie danych (przykład)

```python
import json

with open("data/dataset.json", encoding="utf-8") as f:
    dataset = json.load(f)

# lub przez pandas
import pandas as pd
df = pd.read_json("data/dataset.json")
```

## Struktura projektu

```
data/                        # pobrane i przetworzone dane
  Adam_Mickiewicz/           # surowe pliki tekstowe
  Juliusz_Słowacki/
  Zygmunt_Krasiński/
  dataset.json               # gotowy zbiór danych
src/nlp_stylometry/
  api/fetch_data.py          # pobieranie z wolnelektury.pl
  config.py                  # konfiguracja (URL API, ścieżki)
  preprocessing.py           # czyszczenie tekstu podczas pobierania
  preprocess.py              # filtrowanie, segmentacja, balansowanie
  main.py                    # główny pipeline
```
