from collections.abc import Collection

from src.common.congif import DOMAIN_LENGTH
from src.common.models import LaplaceResultTypes

import numpy.typing as npt
import numpy as np


def solve_laplace(
    x: float | npt.NDArray[np.float64], y: float | npt.NDArray[np.float64]
) -> float | npt.NDArray[np.float64]:
    """
    Return the exact solution u(x, y).

    This function represents the real part of the complex-valued function
    f(z) introduced by the user.
    """
    return np.exp(x) * np.cos(y)


def rozwiaz_laplace(
    rozmiar_siatki: int,
    tolerancja: float,
    max_iteracji: int,
    iteracje_do_zapisu: Collection[int] | None = None,
) -> WynikLaplace:
    """
    Rozwiązuje równanie Laplace'a na kwadracie [0, 1] x [0, 1]
    metodą iteracyjną Gaussa-Seidla.

    Wejście:
    - rozmiar_siatki: liczba punktów siatki w każdym kierunku,
    - tolerancja: warunek stopu iteracji,
    - max_iteracji: maksymalna liczba iteracji,
    - iteracje_do_zapisu: numery iteracji, dla których zapisywane
      są migawki rozwiązania.

    Wyjście:
    - słownik zawierający siatkę, rozwiązanie numeryczne,
      liczbę iteracji, migawki oraz dane do analizy błędu.
    """

    iteracje_zapisu = set(iteracje_do_zapisu or ())

    x = np.linspace(0.0, DLUGOSC_BOKU, rozmiar_siatki)
    y = np.linspace(0.0, DLUGOSC_BOKU, rozmiar_siatki)
    krok_siatki = DLUGOSC_BOKU / (rozmiar_siatki - 1)

    # Konwencja:
    # u[i, j] to nasze u[x_i, y_j], gdzie i, j są odpowiednimi
    # indeksami dla osi X i Y.

    u = np.zeros((rozmiar_siatki, rozmiar_siatki), dtype=float)

    # Warunki brzegowe funkcji dokładnej dla kwadratu [0, A] x [0, A],
    # gdzie A to długość boku.
    # Kolejno: y -> 0, y -> długość boku, x -> 0, x -> długość boku.
    u[:, 0] = u_dokladne(x, 0.0)
    u[:, -1] = u_dokladne(x, DLUGOSC_BOKU)
    u[0, :] = u_dokladne(0.0, y)
    u[-1, :] = u_dokladne(DLUGOSC_BOKU, y)

    migawki: dict[int, npt.NDArray[np.float64]] = {}
    iteracja = 0

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

        if iteracja in iteracje_zapisu:
            migawki[iteracja] = u.copy()

        if max_zmiana < tolerancja:
            print(f"Zbieżność osiągnięta po {iteracja} iteracjach")
            break

    # Gdy pętla skończy się bez break.
    else:
        print("Nie osiągnięto zbieżności w maksymalnej liczbie iteracji")

    # Porównanie metody numerycznej i dokładnej.
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
        "krok_siatki": krok_siatki,
        "iteracje": iteracja,
        "migawki": migawki,
        "u_dokladne": siatka_dokladna,
        "blad": blad,
        "blad_max": blad_max,
        "rozmiar_siatki": rozmiar_siatki,
    }
