import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# Ustawienia / Stałe

DLUGOSC_BOKU = 1.0

ROZMIARY_SIATKI = (10, 20, 40, 80)
TOLERANCJE = {
    10: 1e-10,
    20: 1e-10,
    40: 1e-11,
    80: 1e-11,
}

MAX_ITERACJI = 50000

N_SIATKA_WYKRESY = 81
TOLERANCJA_WYKRESY = 1e-10
MAX_ITERACJI_WYKRESY = 30000

POZIOMY_Y_DO_WYKRESU = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90)
ITERACJE_DO_WYKRESU = (20, 100, 200, 2000)


# Badana funkcja


def u_dokladne(
    x: float | npt.NDArray[np.float64], y: float | npt.NDArray[np.float64]
) -> float | npt.NDArray[np.float64]:
    return np.exp(x) * np.cos(y)


# Metoda rozwiązująca równanie Laplace'a metodą relaksacyjną


def rozwiaz_laplace(rozmiar_siatki, tolerancja, max_iteracji, iteracje_do_zapisu=None):
    if iteracje_do_zapisu is None:
        iteracje_do_zapisu = []

    x = np.linspace(0.0, DLUGOSC_BOKU, rozmiar_siatki)
    y = np.linspace(0.0, DLUGOSC_BOKU, rozmiar_siatki)
    krok_siatki = DLUGOSC_BOKU / (rozmiar_siatki - 1)

    # Konwencja:
    # u[i, j] to jest nasze u[x_i, y_j], i,j - odpowiednie indeksy dla osi X i Y

    u = np.zeros((rozmiar_siatki, rozmiar_siatki), dtype=float)
