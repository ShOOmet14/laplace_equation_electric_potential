def rozwiaz_laplace_relaksacja(
    n_siatki,
    tolerancja,
    maks_iteracji,
    iteracje_do_zapisu=None,
    verbose=False,
):
    if iteracje_do_zapisu is None:
        iteracje_do_zapisu = []

    x = np.linspace(0.0, A, n_siatki)
    y = np.linspace(0.0, A, n_siatki)
    h = A / (n_siatki - 1)

    # Konwencja:
    # u[i, j] ~ u(x_i, y_j)
    u = np.zeros((n_siatki, n_siatki), dtype=float)

    # Warunki brzegowe z funkcji dokladnej
    u[:, 0] = u_dokladne(x, 0.0)  # y = 0
    u[:, -1] = u_dokladne(x, A)  # y = A
    u[0, :] = u_dokladne(0.0, y)  # x = 0
    u[-1, :] = u_dokladne(A, y)  # x = A

    migawki = {}
    iteracja = 0

    for iteracja in range(1, maks_iteracji + 1):
        maks_zmiana = 0.0

        for i in range(1, n_siatki - 1):
            for j in range(1, n_siatki - 1):
                stara_wartosc = u[i, j]

                nowa_wartosc = 0.25 * (
                    u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1]
                )

                u[i, j] = nowa_wartosc

                zmiana = abs(nowa_wartosc - stara_wartosc)
                if zmiana > maks_zmiana:
                    maks_zmiana = zmiana

        # Ponowne narzucenie warunkow brzegowych
        u[:, 0] = u_dokladne(x, 0.0)
        u[:, -1] = u_dokladne(x, A)
        u[0, :] = u_dokladne(0.0, y)
        u[-1, :] = u_dokladne(A, y)

        if iteracja in iteracje_do_zapisu:
            migawki[iteracja] = u.copy()

        if verbose and (iteracja % 100 == 0 or iteracja == 1):
            print(f"iteracja = {iteracja}, maks_zmiana = {maks_zmiana:.8e}")

        if maks_zmiana < tolerancja:
            if verbose:
                print(f"Zbieznosc osiagnieta po {iteracja} iteracjach")
            break
    else:
        if verbose:
            print("Nie osiagnieto zbieznosci w zadanej liczbie iteracji")

    migawki["koniec"] = u.copy()

    X, Y = np.meshgrid(x, y, indexing="ij")
    siatka_dokladna = u_dokladne(X, Y)
    blad = np.abs(u - siatka_dokladna)
    blad_maksymalny = np.max(blad)

    return {
        "x": x,
        "y": y,
        "u": u,
        "h": h,
        "iteracje": iteracja,
        "migawki": migawki,
        "u_dokladne": siatka_dokladna,
        "blad": blad,
        "blad_maksymalny": blad_maksymalny,
        "N": n_siatki,
    }


# ============================================================
# BADANIE WPLYWU GESTOSCI SIATKI
# ============================================================


def uruchom_badanie_siatki():
    wyniki = []

    print("\nBADANIE SIATKI I DOKLADNOSCI")
    print(f"{'N':>6} {'h':>10} {'iteracje':>12} {'blad_maks':>15}")

    for n_siatki in ROZMIARY_SIATKI:
        wynik = rozwiaz_laplace_relaksacja(
            n_siatki=n_siatki,
            tolerancja=TOLERANCJE[n_siatki],
            maks_iteracji=MAKS_ITERACJI_BADANIE,
            iteracje_do_zapisu=[],
            verbose=False,
        )

        wyniki.append(
            {
                "N": wynik["N"],
                "h": wynik["h"],
                "iteracje": wynik["iteracje"],
                "blad_maksymalny": wynik["blad_maksymalny"],
            }
        )

        print(
            f"{wynik['N']:>6}"
            f"{wynik['h']:>10.5f}"
            f"{wynik['iteracje']:>12}"
            f"{wynik['blad_maksymalny']:>15.8e}"
        )

    print("\nPrzyblizony rzad dokladnosci p:")
    for k in range(len(wyniki) - 1):
        e1 = wyniki[k]["blad_maksymalny"]
        e2 = wyniki[k + 1]["blad_maksymalny"]
        p = np.log(e1 / e2) / np.log(2.0)

        print(f"N={wyniki[k]['N']:>3} -> N={wyniki[k + 1]['N']:>3}: p = {p:.4f}")

    lepsze_niz_cel = [w for w in wyniki if w["blad_maksymalny"] < 1e-4]
    if lepsze_niz_cel:
        pierwszy_dobry = lepsze_niz_cel[0]
        print(
            f"\nDokladnosc lepsza od 1e-4 osiagnieta dla:"
            f" N = {pierwszy_dobry['N']}, "
            f"blad_maks = {pierwszy_dobry['blad_maksymalny']:.8e}"
        )
    else:
        print("\nNie osiagnieto poziomu 1e-4 dla zadanych siatek.")

    return wyniki


# ============================================================
# WYKRES 0 - SAMA FUNKCJA BADANA
# ============================================================


def rysuj_przekroje_dokladne(wynik):
    x = wynik["x"]
    y = wynik["y"]
    u_dokladne_siatka = wynik["u_dokladne"]

    plt.figure(figsize=(11, 7))

    for y_poziom in POZIOMY_Y_DO_WYKRESU:
        j = int(np.argmin(np.abs(y - y_poziom)))
        plt.plot(
            x,
            u_dokladne_siatka[:, j],
            linewidth=2.5,
            label=f"u(x, {y[j]:.2f})",
        )

    plt.xlabel("x")
    plt.ylabel("u(x, y)")
    plt.title("Funkcja badana: u(x,y) = exp(x) cos(y) - przekroje dokladne")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("funkcja_badana_przekroje.png", dpi=200)
    plt.show()


# ============================================================
# WYKRESY 2D DLA WYBRANYCH ITERACJI
# ============================================================


def rysuj_przekroje_dla_iteracji(wynik):
    x = wynik["x"]
    y = wynik["y"]
    u_dokladne_siatka = wynik["u_dokladne"]
    migawki = wynik["migawki"]

    for liczba_iteracji in ITERACJE_DO_WYKRESU:
        if liczba_iteracji not in migawki:
            continue

        u_przyblizone = migawki[liczba_iteracji]

        plt.figure(figsize=(11, 7))

        for y_poziom in POZIOMY_Y_DO_WYKRESU:
            j = int(np.argmin(np.abs(y - y_poziom)))

            (linia_dokladna,) = plt.plot(
                x,
                u_dokladne_siatka[:, j],
                linewidth=2.0,
                label=f"dokladne, y = {y[j]:.2f}",
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
        plt.title(f"Przekroje rozwiazania po {liczba_iteracji} iteracjach")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        plt.savefig(f"przekroje_iteracja_{liczba_iteracji}.png", dpi=200)

        plt.show()


# ============================================================
# WYKRES FINALNY - PO ZBIEZNOSCI
# ============================================================


def rysuj_przekroje_koncowe(wynik):
    x = wynik["x"]
    y = wynik["y"]
    u_dokladne_siatka = wynik["u_dokladne"]
    u_koncowe = wynik["migawki"]["koniec"]

    plt.figure(figsize=(11, 7))

    for y_poziom in POZIOMY_Y_DO_WYKRESU:
        j = int(np.argmin(np.abs(y - y_poziom)))

        (linia_dokladna,) = plt.plot(
            x,
            u_dokladne_siatka[:, j],
            linewidth=2.0,
            label=f"dokladne, y = {y[j]:.2f}",
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
    plt.title("Przekroje rozwiazania po zbieznosci")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig("przekroje_iteracja_koniec.png", dpi=200)

    plt.show()


# ============================================================
# WYKRES 3D - KONCOWE ROZWIAZANIE NUMERYCZNE
# ============================================================


def rysuj_rozwiazanie_3d(wynik):
    x = wynik["x"]
    y = wynik["y"]
    u_num = wynik["u"]

    X, Y = np.meshgrid(x, y, indexing="ij")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    powierzchnia = ax.plot_surface(
        X,
        Y,
        u_num,
        cmap="viridis",
        edgecolor="none",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x, y)")
    ax.set_title("Rozwiazanie numeryczne rownania Laplace'a - wykres 3D")

    fig.colorbar(powierzchnia, ax=ax, shrink=0.75, pad=0.1)
    plt.tight_layout()
    plt.savefig("rozwiazanie_3d.png", dpi=200)

    plt.show()


# ============================================================
# URUCHOMIENIE
# ============================================================

if __name__ == "__main__":
    # 1. Badanie siatki i dokladnosci
    wyniki_badania = uruchom_badanie_siatki()

    # 2. Jeden przypadek do wykresow
    wynik_do_wykresow = rozwiaz_laplace_relaksacja(
        n_siatki=N_SIATKA_WYKRESY,
        tolerancja=TOLERANCJA_WYKRESY,
        maks_iteracji=MAKS_ITERACJI_WYKRESY,
        iteracje_do_zapisu=ITERACJE_DO_WYKRESU,
        verbose=True,
    )

    print("\nWYNIK DLA WYKRESOW")
    print(f"N = {wynik_do_wykresow['N']}")
    print(f"h = {wynik_do_wykresow['h']}")
    print(f"iteracje = {wynik_do_wykresow['iteracje']}")
    print(f"blad_maksymalny = {wynik_do_wykresow['blad_maksymalny']:.8e}")

    # 3. Wykresy
    rysuj_przekroje_dokladne(wynik_do_wykresow)
    rysuj_przekroje_dla_iteracji(wynik_do_wykresow)
    rysuj_przekroje_koncowe(wynik_do_wykresow)
    rysuj_rozwiazanie_3d(wynik_do_wykresow)
