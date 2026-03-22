from collections.abc import Collection
from typing import TypedDict

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


# Część rzeczywista badanej funkcji zespolonej, tutaj f(z) = exp(z),
# gdzie z = x + iy


def u_dokladne(
    x: float | npt.NDArray[np.float64], y: float | npt.NDArray[np.float64]
) -> float | npt.NDArray[np.float64]:
    return np.exp(x) * np.cos(y)


# Funkcja rozwiązująca równanie Laplace'a metodą relaksacyjną
class WynikLaplace(TypedDict):
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    u: npt.NDArray[np.float64]
    h: float
    iteracje: int
    migawki: dict[int, npt.NDArray[np.float64]]
    u_dokladne: npt.NDArray[np.float64]
    blad: npt.NDArray[np.float64]
    blad_maksymalny: float
    rozmiar_siatki: int


def rozwiaz_laplace(
    rozmiar_siatki: int,
    tolerancja: float,
    max_iteracji: int,
    iteracje_do_zapisu: Collection[int] | None = None,
) -> WynikLaplace:
    iteracje_zapisu = set(iteracje_do_zapisu or ())

    x = np.linspace(0.0, DLUGOSC_BOKU, rozmiar_siatki)
    y = np.linspace(0.0, DLUGOSC_BOKU, rozmiar_siatki)
    krok_siatki = DLUGOSC_BOKU / (rozmiar_siatki - 1)

    # Konwencja:
    # u[i, j] to jest nasze u[x_i, y_j], i,j - odpowiednie indeksy dla osi X i Y

    u = np.zeros((rozmiar_siatki, rozmiar_siatki), dtype=float)

    # Warunki brzegowe funkcji dokładnej, dla kwadratu [0, A] x [0, A]
    # gdzie A to długość boku
    # Kolejno: y -> 0, y -> długość boku, x -> 0, x -> długość boku
    u[:, 0] = u_dokladne(x, 0.0)
    u[:, -1] = u_dokladne(x, DLUGOSC_BOKU)
    u[0, :] = u_dokladne(0.0, y)
    u[-1, :] = u_dokladne(DLUGOSC_BOKU, y)

    migawki: dict[int, npt.NDArray[np.float64]] = {}
    iteracja = 0

    # Metoda iteracyjna Gaussa-Seidla, zamiast Jacobiego
    for iteracja in range(1, max_iteracji + 1):
        max_zmiana = 0.0

        for i in range(1, rozmiar_siatki - 1):
            for j in range(1, rozmiar_siatki - 1):
                stara_wartosc = u[i, j]

                nowa_wartosc = 0.25 * (
                    u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1]
                )

                u[i, j] = nowa_wartosc

                zmiana = abs(nowa_wartosc - stara_wartosc)
                if zmiana > max_zmiana:
                    max_zmiana = zmiana

        """
        u[:, 0] = u_dokladne(x, 0.0)
        u[:, -1] = u_dokladne(x, DLUGOSC_BOKU)
        u[0, :] = u_dokladne(0.0, y)
        u[-1, :] = u_dokladne(DLUGOSC_BOKU, y)
        """

        if iteracja in iteracje_zapisu:
            migawki[iteracja] = u.copy()

        if max_zmiana < tolerancja:
            print(f"Zbieżność osiągnięta po {iteracja} iteracjach")
            break

    # Gdy pętla skończy się bez break
    else:
        print("Nie osiągnięto zbieżności w maksymalnej liczbie iteracji")

    # Porównanie metody numerycznej i dokładnej
    x_nowe, y_nowe = np.meshgrid(x, y, indexing="ij")
    siatka_dokladna: npt.NDArray[np.float64] = np.asarray(
        u_dokladne(x_nowe, y_nowe), dtype=np.float64
    )
    blad = np.abs(u - siatka_dokladna)
    blad_max = float(np.max(blad))

    return {
        "x": x,
        "y": y,
        "u": u,
        "h": krok_siatki,
        "iteracje": iteracja,
        "migawki": migawki,
        "u_dokladne": siatka_dokladna,
        "blad": blad,
        "blad_maksymalny": blad_max,
        "rozmiar_siatki": rozmiar_siatki,
    }
