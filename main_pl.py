from collections.abc import Collection
from typing import TypedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def u_dokladne(
    x: float | npt.NDArray[np.float64], y: float | npt.NDArray[np.float64]
) -> float | npt.NDArray[np.float64]:
    """
    Zwraca rozwiązanie dokładne u(x, y).

    Funkcja ta jest częścią rzeczywistą funkcji zespolonej f(z)
    wprowadzonej przez użytkownika.
    """
    return np.exp(x) * np.cos(y)


class WynikLaplace(TypedDict):
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    u: npt.NDArray[np.float64]
    krok_siatki: float
    iteracje: int
    migawki: dict[int, npt.NDArray[np.float64]]
    u_dokladne: npt.NDArray[np.float64]
    blad: npt.NDArray[np.float64]
    blad_max: float
    rozmiar_siatki: int


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


class RekordBadania(TypedDict):
    rozmiar_siatki: int
    krok_siatki: float
    iteracje: int
    blad_max: float


def badanie_siatki() -> list[RekordBadania]:
    """
    Uruchamia obliczenia dla kilku rozmiarów siatki i porównuje dokładność.

    Działanie funkcji:
    - wywołuje funkcję rozwiaz_laplace() dla kolejnych rozmiarów siatki,
    - zbiera liczbę iteracji, krok siatki i błąd maksymalny,
    - wypisuje przybliżony rząd dokładności,
    - sprawdza, czy osiągnięto poziom błędu 1e-4.

    Wyjście:
    - lista słowników z wynikami dla kolejnych siatek.
    """

    wyniki: list[RekordBadania] = []

    print("\nBADANIE SIATKI I DOKŁADNOŚCI")
    print(f"{'Rozmiar':>6} {'Krok siatki':>10} {'Iteracje':>10} {'Błąd max':>15}")

    for rozmiar_siatki in ROZMIARY_SIATKI:
        wynik = rozwiaz_laplace(
            rozmiar_siatki=rozmiar_siatki,
            tolerancja=TOLERANCJE[rozmiar_siatki],
            max_iteracji=MAX_ITERACJI,
        )

        rekord: RekordBadania = {
            "rozmiar_siatki": wynik["rozmiar_siatki"],
            "krok_siatki": wynik["krok_siatki"],
            "iteracje": wynik["iteracje"],
            "blad_max": wynik["blad_max"],
        }

        wyniki.append(rekord)

        print(
            f"{wynik['rozmiar_siatki']:>6}"
            f"{wynik['krok_siatki']:>10.5f}"
            f"{wynik['iteracje']:>12}"
            f"{wynik['blad_max']:>15.8e}"
        )

    print("\nPrzybliżony rząd dokładności p:")

    for k in range(len(wyniki) - 1):
        krok_siatki_1 = wyniki[k]["krok_siatki"]
        krok_siatki_2 = wyniki[k + 1]["krok_siatki"]
        blad_max_1 = wyniki[k]["blad_max"]
        blad_max_2 = wyniki[k + 1]["blad_max"]

        p = np.log(blad_max_1 / blad_max_2) / np.log(krok_siatki_1 / krok_siatki_2)

        print(
            f"N = {wyniki[k]['rozmiar_siatki']:>3} -> "
            f"N = {wyniki[k + 1]['rozmiar_siatki']:>3}: p = {p:.4f}"
        )

    pierwszy_dobry = next(
        (wynik for wynik in wyniki if wynik["blad_max"] < 1e-4),
        None,
    )

    if pierwszy_dobry is not None:
        print(
            f"\nDokładność lepsza od 1e-4 osiągnięta dla: "
            f"Rozmiar siatki = {pierwszy_dobry['rozmiar_siatki']}, "
            f"Błąd max = {pierwszy_dobry['blad_max']:.8e}"
        )
    else:
        print("\nNie osiągnięto poziomu 1e-4 dla zadanych siatek.")

    return wyniki


def rysuj_przekroje(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    wartosci_siatka: npt.NDArray[np.float64],
) -> None:
    """
    Rysuje przekroje funkcji dla wybranych poziomów y, zapisuje wykres
    i wyświetla go.

    Wejście:
    - x: współrzędne siatki w kierunku osi X,
    - y: współrzędne siatki w kierunku osi Y,
    - wartosci_siatka: macierz wartości funkcji na siatce.

    Wyjście:
    - nic.
    """

    plt.figure(figsize=(11, 7))

    for y_poziom in POZIOMY_Y_DO_WYKRESU:
        # Znajdowanie indeksu tablicy y, gdzie wartość jest najbliższa
        # danemu poziomowi Y.
        j = int(np.argmin(np.abs(y - y_poziom)))

        plt.plot(
            x,
            wartosci_siatka[:, j],
            linewidth=2.5,
            label=f"y = {y[j]:.2f}",
        )

    plt.xlabel("x")
    plt.ylabel("u(x, y)")
    plt.title("Przekroje funkcji na wybranych poziomach y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(KATALOG_WYKRESOW / "przekroje.png", dpi=200)
    plt.show()


def rysuj_przekroje_dla_iteracji(wynik: WynikLaplace) -> None:
    """
    Funkcja tworzy, zapisuje i wyświetla wykresy przekrojów.

    Parametry:
    - wynik: wynik obliczeń zawierający x, y, rozwiązanie dokładne
      i migawki kolejnych przybliżeń.

    Wyjście:
    - nic.
    """

    x = wynik["x"]
    y = wynik["y"]
    u_dokladne_siatka = wynik["u_dokladne"]
    migawki = wynik["migawki"]

    indeksy_poziomow_y: list[int] = []
    for y_poziom in POZIOMY_Y_DO_WYKRESU:
        j = int(np.argmin(np.abs(y - y_poziom)))
        indeksy_poziomow_y.append(j)

    for iteracja in ITERACJE_DO_WYKRESU:
        if iteracja not in migawki:
            continue

        u_przyblizone = migawki[iteracja]

        plt.figure(figsize=(11, 7))

        for j in indeksy_poziomow_y:
            (linia_dokladna,) = plt.plot(
                x,
                u_dokladne_siatka[:, j],
                linewidth=2.0,
                label=f"dokładne, y = {y[j]:.2f}",
            )

            kolor = linia_dokladna.get_color()

            plt.plot(
                x,
                u_przyblizone[:, j],
                "o--",
                color=kolor,
                linewidth=1.2,
                markersize=3,
                markevery=2,
                label=f"relaksacja, y = {y[j]:.2f}",
            )

        plt.xlabel("x")
        plt.ylabel("u(x, y)")
        plt.title(f"Przekroje rozwiązania po {iteracja} iteracjach")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        plt.savefig(KATALOG_WYKRESOW / f"przekroje_iteracja_{iteracja}.png", dpi=200)

        plt.show()


def rysuj_przekroje_koncowe(wynik: WynikLaplace) -> None:
    """
    Rysuje przekroje końcowego rozwiązania numerycznego, zapisuje
    i wyświetla obraz.

    Wejście:
    - wynik: wynik obliczeń zawierający siatkę x, y, rozwiązanie
      dokładne oraz końcowe rozwiązanie numeryczne.

    Wyjście:
    - nic.
    """
    x = wynik["x"]
    y = wynik["y"]
    u_dokladne_siatka = wynik["u_dokladne"]
    u_koncowe = wynik["u"]

    plt.figure(figsize=(11, 7))

    for y_poziom in POZIOMY_Y_DO_WYKRESU:
        j = int(np.argmin(np.abs(y - y_poziom)))

        (linia_dokladna,) = plt.plot(
            x,
            u_dokladne_siatka[:, j],
            linewidth=2.0,
            label=f"dokładne, y = {y[j]:.2f}",
        )

        kolor = linia_dokladna.get_color()

        plt.plot(
            x,
            u_koncowe[:, j],
            "o--",
            color=kolor,
            linewidth=1.2,
            markersize=3,
            markevery=2,
            label=f"relaksacja, y = {y[j]:.2f}",
        )

    plt.xlabel("x")
    plt.ylabel("u(x, y)")
    plt.title("Przekroje rozwiązania po zbieżności")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(KATALOG_WYKRESOW / "przekroje_iteracja_koniec.png", dpi=200)
    plt.show()


def rysuj_rozwiazanie_3d(wynik: WynikLaplace) -> None:
    """
    Rysuje wykres 3D końcowego rozwiązania numerycznego, zapisuje go
    i wyświetla.

    Wejście:
    - wynik: wynik obliczeń zawierający siatkę x, y oraz końcowe
      rozwiązanie numeryczne u.

    Wyjście:
    - nic.
    """
    x = wynik["x"]
    y = wynik["y"]
    u_numeryczne = wynik["u"]

    X, Y = np.meshgrid(x, y, indexing="ij")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    powierzchnia = ax.plot_surface(
        X,
        Y,
        u_numeryczne,
        cmap="viridis",
        edgecolor="none",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x, y)")
    ax.set_title("Rozwiązanie numeryczne równania Laplace'a - wykres 3D")

    fig.colorbar(powierzchnia, ax=ax, shrink=0.75, pad=0.1)
    plt.tight_layout()
    plt.savefig(KATALOG_WYKRESOW / "rozwiazanie_3d.png", dpi=200)

    plt.show()


if __name__ == "__main__":
    wyniki_badania = badanie_siatki()

    wynik_do_wykresow = rozwiaz_laplace(
        rozmiar_siatki=N_SIATKA_WYKRESY,
        tolerancja=TOLERANCJA_WYKRESY,
        max_iteracji=MAX_ITERACJI_WYKRESY,
        iteracje_do_zapisu=ITERACJE_DO_WYKRESU,
    )

    print("\nWYNIK DLA WYKRESÓW")
    print(f"Rozmiar siatki = {wynik_do_wykresow['rozmiar_siatki']}")
    print(f"Krok siatki = {wynik_do_wykresow['krok_siatki']}")
    print(f"Liczba iteracji = {wynik_do_wykresow['iteracje']}")
    print(f"Błąd max = {wynik_do_wykresow['blad_max']:.8e}")

    rysuj_przekroje(
        x=wynik_do_wykresow["x"],
        y=wynik_do_wykresow["y"],
        wartosci_siatka=wynik_do_wykresow["u_dokladne"],
    )
    rysuj_przekroje_dla_iteracji(wynik_do_wykresow)
    rysuj_przekroje_koncowe(wynik_do_wykresow)
    rysuj_rozwiazanie_3d(wynik_do_wykresow)
