import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, RadioButtons, CheckButtons
from matplotlib.patches import Circle


# ============================================================
# FUNÇÕES FÍSICAS
# ============================================================

def lj_potential(r, sigma, epsilon):
    """Potencial de Lennard-Jones."""
    return 4.0 * epsilon * (
        (sigma / r)**12
        - (sigma / r)**6
    )


def lj_force(r, sigma, epsilon):
    """Força radial de Lennard-Jones."""
    return (48.0 * epsilon / sigma) * (
        (sigma / r)**13
        - 0.5 * (sigma / r)**7
    )


def wca_potential(r, sigma, epsilon):
    """Potencial WCA."""
    rc = 2**(1/6) * sigma

    r = np.asarray(r)

    return np.where(
        r < rc,
        lj_potential(r, sigma, epsilon) + epsilon,
        0.0
    )


def wca_force(r, sigma, epsilon):
    """Força WCA."""
    rc = 2**(1/6) * sigma

    r = np.asarray(r)

    return np.where(
        r < rc,
        lj_force(r, sigma, epsilon),
        0.0
    )


# ============================================================
# PARÂMETROS INICIAIS
# ============================================================

sigma0 = 1.0
epsilon0 = 1.0
rij0 = 1.5


# ============================================================
# INTERVALO FIXO
# ============================================================

r_min = 0.6
r_max = 4.0

r = np.linspace(
    r_min,
    r_max,
    3000
)


# ============================================================
# FIGURA
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 6.5)
)

plt.subplots_adjust(
    left=0.10,
    right=0.72,
    bottom=0.30
)


# ============================================================
# PAINEL DOS ÁTOMOS
# ============================================================

ax_atoms = fig.add_axes(
    [0.76, 0.08, 0.22, 0.28]
)

ax_atoms.set_aspect("equal")

ax_atoms.set_xlim(
    -1.2,
    5.0
)

ax_atoms.set_ylim(
    -1.6,
    2.5
)

ax_atoms.set_xticks([])
ax_atoms.set_yticks([])

ax_atoms.set_title(
    r"Separação interatômica $|r_{ij}|$",
    fontsize=11
)


# ============================================================
# FUNÇÃO QUE DESENHA OS ÁTOMOS
# ============================================================

def draw_atoms(sigma,epsilon, rij, selected):

    ax_atoms.clear()

    ax_atoms.set_aspect("equal")

    ax_atoms.set_xlim(
        -1.2,
        5.0
    )

    ax_atoms.set_ylim(
        -1.3,
        1.3
    )

    ax_atoms.set_xticks([])
    ax_atoms.set_yticks([])

    ax_atoms.set_title(
        r"Separação interatômica $|r_{ij}|$",
        fontsize=11
    )

    # ========================================================
    # TAMANHO VISUAL DOS ÁTOMOS
    #
    # sigma funciona como uma escala de tamanho.
    # Usamos aproximadamente sigma/2 como raio visual.
    # ========================================================

    radius = sigma / 2


    # ========================================================
    # POSIÇÕES DOS CENTROS
    # ========================================================

    x1 = 0.0
    x2 = rij

    y1 = 0.0
    y2 = 0.0

    # ========================================================
    # VALOR DA FUNÇÃO NO r_ij ATUAL
    # ========================================================

    if selected == "Potencial LJ":

        value = lj_potential(
            rij,
            sigma,
            epsilon
        )

        value_text = rf"$U_{{LJ}}(r_{{ij}}) = {value:.3f}$"


    elif selected == "Força LJ":

        value = lj_force(
            rij,
            sigma,
            epsilon
        )

        value_text = rf"$F_{{LJ}}(r_{{ij}}) = {value:.3f}$"


    elif selected == "Potencial WCA":

        value = wca_potential(
            np.array([rij]),
            sigma,
            epsilon
        )[0]

        value_text = rf"$U_{{WCA}}(r_{{ij}}) = {value:.3f}$"


    elif selected == "Força WCA":

        value = wca_force(
            np.array([rij]),
            sigma,
            epsilon
        )[0]

        value_text = rf"$F_{{WCA}}(r_{{ij}}) = {value:.3f}$"

    # ========================================================
    # ÁTOMO i
    # ========================================================

    atom_i = Circle(
        (x1, y1),
        radius,
        facecolor="lightgray",
        edgecolor="black",
        linewidth=1.5,
        zorder=3
    )

    ax_atoms.add_patch(atom_i)


    # ========================================================
    # ÁTOMO j
    # ========================================================

    atom_j = Circle(
        (x2, y2),
        radius,
        facecolor="silver",
        edgecolor="black",
        linewidth=1.5,
        zorder=3
    )

    ax_atoms.add_patch(atom_j)


    # ========================================================
    # CENTROS
    # ========================================================

    ax_atoms.plot(
        x1,
        y1,
        marker=".",
        color="black",
        markersize=6,
        zorder=4
    )

    ax_atoms.plot(
        x2,
        y2,
        marker=".",
        color="black",
        markersize=6,
        zorder=4
    )


    # ========================================================
    # LINHA ENTRE OS CENTROS
    # ========================================================

    ax_atoms.plot(
        [x1, x2],
        [0, 0],
        linestyle="--",
        color="black",
        linewidth=1,
        zorder=1
    )


    # ========================================================
    # INDICAÇÃO DA DISTÂNCIA
    # ========================================================

    y_arrow = -0.95

    ax_atoms.annotate(
        "",
        xy=(x2, y_arrow),
        xytext=(x1, y_arrow),

        arrowprops=dict(
            arrowstyle="<->",
            linewidth=1.4
        )
    )

    ax_atoms.text(
        (x1 + x2) / 2,
        y_arrow - 0.13,

        rf"$|r_{{ij}}|={rij:.2f}$",

        horizontalalignment="center",
        verticalalignment="top",
        fontsize=10
    )


    # ========================================================
    # IDENTIFICAÇÃO DOS ÁTOMOS
    # ========================================================

    ax_atoms.text(
        x1,
        0,
        r"$i$",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=12,
        fontweight="bold"
    )

    ax_atoms.text(
        x2,
        0,
        r"$j$",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=12,
        fontweight="bold"
    )
    ax_atoms.text(
        0.5,
        0.92,
        value_text,

        transform=ax_atoms.transAxes,

        horizontalalignment="center",
        verticalalignment="top",

        fontsize=11,

        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="gray",
            alpha=0.85
        )
    )

# ============================================================
# FUNÇÃO PRINCIPAL DE ATUALIZAÇÃO
# ============================================================

def update_plot():

    sigma = sigma_slider.val
    epsilon = epsilon_slider.val
    rij = rij_slider.val

    rc = 2**(1/6) * sigma

    selected = plot_selector.value_selected

    ax.clear()


    # ========================================================
    # POTENCIAL LJ
    # ========================================================

    if selected == "Potencial LJ":

        y = lj_potential(
            r,
            sigma,
            epsilon
        )

        ax.plot(
            r,
            y,
            linewidth=2.2,
            color="black",
            label=r"$U_{\mathrm{LJ}}(r)$"
        )

        ax.set_ylabel(
            r"$U(r)$",
            fontsize=13
        )

        ax.set_title(
            "Potencial de Lennard-Jones",
            fontsize=14
        )

        ax.set_ylim(
            -3,
            15
        )


    # ========================================================
    # FORÇA LJ
    # ========================================================

    elif selected == "Força LJ":

        y = lj_force(
            r,
            sigma,
            epsilon
        )

        ax.plot(
            r,
            y,
            linewidth=2.2,
            color="black",
            label=r"$F_{\mathrm{LJ}}(r)$"
        )

        ax.set_ylabel(
            r"$F(r)$",
            fontsize=13
        )

        ax.set_title(
            "Força de Lennard-Jones",
            fontsize=14
        )

        ax.set_ylim(
            -10,
            80
        )


    # ========================================================
    # POTENCIAL WCA
    # ========================================================

    elif selected == "Potencial WCA":

        y = wca_potential(
            r,
            sigma,
            epsilon
        )

        ax.plot(
            r,
            y,
            linewidth=2.2,
            color="black",
            label=r"$U_{\mathrm{WCA}}(r)$"
        )

        ax.set_ylabel(
            r"$U(r)$",
            fontsize=13
        )

        ax.set_title(
            "Potencial repulsivo WCA",
            fontsize=14
        )

        ax.set_ylim(
            -1,
            15
        )


    # ========================================================
    # FORÇA WCA
    # ========================================================

    else:

        y = wca_force(
            r,
            sigma,
            epsilon
        )

        ax.plot(
            r,
            y,
            linewidth=2.2,
            color="black",
            label=r"$F_{\mathrm{WCA}}(r)$"
        )

        ax.set_ylabel(
            r"$F(r)$",
            fontsize=13
        )

        ax.set_title(
            "Força repulsiva WCA",
            fontsize=14
        )

        ax.set_ylim(
            -10,
            80
        )


    # ========================================================
    # REGIÕES REPULSIVA E ATRATIVA
    # ========================================================

    if region_checkbox.get_status()[0]:

        # Região repulsiva
        ax.axvspan(
            xmin=r_min,
            xmax=rc,
            color="red",
            alpha=0.10,
            label="Região repulsiva"
        )

        # Região atrativa
        if selected in [
            "Potencial LJ",
            "Força LJ"
        ]:

            ax.axvspan(
                xmin=rc,
                xmax=r_max,
                color="blue",
                alpha=0.08,
                label="Região atrativa"
            )


    # ========================================================
    # LINHAS CARACTERÍSTICAS
    # ========================================================

    # r = sigma
    ax.axvline(
        sigma,
        linestyle=":",
        linewidth=1.4,
        color="gray",
        label=rf"$\sigma={sigma:.2f}$"
    )


    # mínimo LJ / cutoff WCA
    ax.axvline(
        rc,
        linestyle="--",
        linewidth=1.4,
        color="red",
        label=rf"$r_c={rc:.3f}$"
    )


    # zero
    ax.axhline(
        0,
        color="black",
        linewidth=0.8,
        alpha=0.5
    )


    # ========================================================
    # POSIÇÃO ATUAL DE r_ij NO GRÁFICO
    # ========================================================

    ax.axvline(
        rij,
        color="green",
        linestyle="-.",
        linewidth=1.4,
        alpha=0.8,
        label=rf"$|r_{{ij}}|={rij:.2f}$"
    )


    # ========================================================
    # CONFIGURAÇÃO DOS EIXOS
    # ========================================================

    ax.set_xlim(
        r_min,
        r_max
    )

    ax.set_xlabel(
        r"Distância $|r_{ij}|$",
        fontsize=13
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        loc="upper right",
        fontsize=8
    )


    # ========================================================
    # ATUALIZA REPRESENTAÇÃO DOS ÁTOMOS
    # ========================================================

    draw_atoms(
        sigma,
        epsilon,
        rij,
        selected
    )

    fig.canvas.draw_idle()


# ============================================================
# SLIDER SIGMA
# ============================================================

ax_sigma = plt.axes(
    [0.10, 0.18, 0.55, 0.03]
)

sigma_slider = Slider(
    ax=ax_sigma,
    label=r"$\sigma$",
    valmin=0.5,
    valmax=2.0,
    valinit=sigma0,
    valstep=0.02
)


# ============================================================
# SLIDER EPSILON
# ============================================================

ax_epsilon = plt.axes(
    [0.10, 0.12, 0.55, 0.03]
)

epsilon_slider = Slider(
    ax=ax_epsilon,
    label=r"$\epsilon$",
    valmin=0.5,
    valmax=2.0,
    valinit=epsilon0,
    valstep=0.02
)


# ============================================================
# SLIDER r_ij
# ============================================================

ax_rij = plt.axes(
    [0.10, 0.06, 0.55, 0.03]
)

rij_slider = Slider(
    ax=ax_rij,
    label=r"$|r_{ij}|$",
    valmin=r_min,
    valmax=r_max,
    valinit=rij0,
    valstep=0.01
)


# ============================================================
# RADIO BUTTONS
# ============================================================

ax_radio = plt.axes(
    [0.77, 0.60, 0.20, 0.25]
)

plot_selector = RadioButtons(
    ax_radio,
    (
        "Potencial LJ",
        "Força LJ",
        "Potencial WCA",
        "Força WCA"
    ),
    active=1
)


# ============================================================
# CHECKBOX
# ============================================================

ax_check = plt.axes(
    [0.77, 0.47, 0.20, 0.08]
)

region_checkbox = CheckButtons(
    ax_check,
    ["Mostrar regiões"],
    [True]
)


# ============================================================
# EVENTOS
# ============================================================

sigma_slider.on_changed(
    lambda val: update_plot()
)

epsilon_slider.on_changed(
    lambda val: update_plot()
)

rij_slider.on_changed(
    lambda val: update_plot()
)

plot_selector.on_clicked(
    lambda label: update_plot()
)

region_checkbox.on_clicked(
    lambda label: update_plot()
)


# ============================================================
# INICIALIZAÇÃO
# ============================================================

update_plot()

plt.show()