# Proponowana refaktoryzacja projektu

## Nowa struktura

```text
refaktor_projektu/
├── images/
│   ├── en/
│   └── pl/
├── run_en.py
├── run_pl.py
└── src/
    ├── common/
    │   ├── app.py
    │   ├── config.py
    │   ├── models.py
    │   ├── plotting.py
    │   └── solver.py
    ├── en/
    │   ├── labels.py
    │   └── main.py
    └── pl/
        ├── labels.py
        └── main.py
```

## Dlaczego taki podział

- `src/common/` zawiera całą wspólną logikę numeryczną i rysowanie wykresów.
- `src/pl/` i `src/en/` przechowują elementy zależne od języka.
- `images/pl/` i `images/en/` rozdzielają pliki wyjściowe według języka.
- `run_pl.py` i `run_en.py` pozwalają uruchamiać projekt z katalogu głównego.

## Najważniejsza zmiana w `Pathlib`

Zamiast polegać na bieżącym katalogu roboczym, ścieżki są wyliczane względem pliku:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGES_ROOT = PROJECT_ROOT / "images"
```

Dzięki temu zapis wykresów działa poprawnie niezależnie od tego, skąd uruchamiasz program.

## Jak uruchamiać

W katalogu głównym projektu:

```bash
python run_pl.py
python run_en.py
```

## Co przenieść z obecnego pliku

Twój obecny `main_pl.py` został rozbity tak:

- obliczenia -> `src/common/solver.py`
- typy danych -> `src/common/models.py`
- wykresy -> `src/common/plotting.py`
- konfiguracja i ścieżki -> `src/common/config.py`
- teksty PL -> `src/pl/labels.py`
- uruchomienie PL -> `src/pl/main.py` lub `run_pl.py`

## Uwaga praktyczna

Jeżeli później będziesz dopisywać wersję angielską, w większości przypadków wystarczy zmienić lub rozszerzyć pliki w `src/en/` bez dublowania solvera i logiki wykresów.
