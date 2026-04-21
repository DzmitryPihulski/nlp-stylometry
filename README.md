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

1. **Pobieranie** — ściąga teksty z API wolnelektury.pl i zapisuje je w `data/{Autor}/{Rodzaj}/{Gatunek}/`
2. **Przetwarzanie** — filtruje pliki niebędące językiem polskim, usuwa puste pliki, usuwa etykiety postaci i didaskalia z dramatów, dzieli teksty na zdania i tworzy równoważne batche po 100 zdań dla każdego autora.

## Wynik

Plik `data/dataset.json` — lista rekordów gotowych do użycia w modelach:

```json
[
  { "author": "Adam_Mickiewicz", "kind": "Liryka", "genre": "Ballada", "title": "Ballady i romanse", "text": "..." },
  { "author": "Juliusz_Słowacki", "kind": "Dramat", "genre": "Dramat_romantyczny", "title": "Kordian", "text": "..." },
  ...
]
```

Każdy rekord to 100 zdań z jednego dzieła (autor, rodzaj, gatunek, tytuł są jednoznaczne na poziomie rekordu). Liczba rekordów jest zbalansowana — każdy autor ma tyle samo próbek łącznie.

### Pola rekordu

| Pole | Opis |
|------|------|
| `author` | Autor w formacie `Imię_Nazwisko` |
| `kind` | Rodzaj literacki (patrz tabela poniżej) |
| `genre` | Gatunek literacki (patrz tabela poniżej) |
| `title` | Tytuł dzieła z API wolnelektury.pl |
| `text` | 100 zdań połączonych spacją |

### Dostępne wartości kind i genre

Dane obejmują następujące rodzaje i gatunki ze zbioru wolnelektury.pl:

**Rodzaje (`kind`):**
- `Epika`
- `Liryka`
- `Dramat`
- `Epika_Liryka`

**Gatunki (`genre`):**
`Ballada`, `Bajka`, `Dedykacja_Motto`, `Dramat_romantyczny`, `Epos`, `List`, `Motto`, `Oda`, `Odezwa`, `Opowiadanie`, `Pieśń`, `Poemat`, `Poemat_dygresyjny`, `Powieść_poetycka`, `Publicystyka`, `Romans`, `Różne`, `Sonet`, `Tragedia`, `Wiersz`, `powieść_historyczna`

> Nazwy katalogów są sanityzowane: spacje → `_`, przecinki usuwane. Oryginalne wartości z API: `Epika, Liryka` → katalog `Epika_Liryka`.

## Wczytanie danych (przykład)

```python
import json

with open("data/dataset.json", encoding="utf-8") as f:
    dataset = json.load(f)

# lub przez pandas
import pandas as pd
df = pd.read_json("data/dataset.json")

# filtrowanie po rodzaju/gatunku/tytule
ballady = [r for r in dataset if r["genre"] == "Ballada"]
dramaty = df[df["kind"] == "Dramat"]
dziady = df[df["title"] == "Dziady"]
```

## Struktura projektu

```
data/                               # pobrane i przetworzone dane
  Adam_Mickiewicz/
    Liryka/
      Ballada/
        1.txt          # treść
        1.json         # {"title": "Ballady i romanse"}
        2.txt
        2.json
      Sonet/
        1.txt
        1.json
    Dramat/
      Dramat_romantyczny/
        1.txt
        1.json
  Juliusz_Słowacki/
    ...
  Zygmunt_Krasiński/
    ...
  dataset.json                      # gotowy zbiór danych
src/nlp_stylometry/
  api/fetch_data.py                 # pobieranie z wolnelektury.pl
  config.py                         # konfiguracja (URL API, ścieżki)
  preprocessing.py                  # czyszczenie tekstu podczas pobierania
  preprocess.py                     # filtrowanie, segmentacja, balansowanie
  main.py                           # główny pipeline
```
