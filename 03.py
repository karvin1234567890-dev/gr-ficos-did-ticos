import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button


# ============================================================
# CONSTANTES DO MODELO
# ============================================================

DIM = 2
N = 64

sigma = 1.0
epsilon = 1.0
mass = 1.0
kB = 1.0

rng = np.random.default_rng(7)


# ============================================================
# PARÂMETROS INICIAIS
# ============================================================

rho0 = 0.80
T00 = 1.00
dt0 = 0.005
rc0 = 2.50
spf0 = 5


# ============================================================
# ESTADO GLOBAL
# ============================================================

running = False
time = 0.0

r = None
v = None
F = None

U = 0.0
virial = 0.0
L = 0.0
V = 0.0

t_hist = []
K_hist = []
U_hist = []
E_hist = []
T_hist = []
P_hist = []

max_history = 1500


# ============================================================
# ESTADO INICIAL
# ============================================================

def initial_positions(rho):

    global L, V

    L = np.sqrt(N / rho)
    V = L**DIM

    n = int(np.sqrt(N))

    a = L / n

    grid = (np.arange(n) + 0.5) * a

    X, Y = np.meshgrid(
        grid,
        grid,
        indexing="ij"
    )

    return np.column_stack(
        (X.ravel(), Y.ravel())
    )


def initial_velocities(T0):

    theta = rng.uniform(
        0,
        2*np.pi,
        N
    )

    speed = np.sqrt(DIM*T0)

    velocities = speed * np.column_stack(
        (
            np.cos(theta),
            np.sin(theta)
        )
    )

    # Remove movimento do centro de massa
    velocities -= velocities.mean(axis=0)

    # Reescala para obter exatamente T0
    T = np.sum(velocities**2) / (DIM*N)

    velocities *= np.sqrt(T0/T)

    return velocities


# ============================================================
# CONDIÇÕES PERIÓDICAS
# ============================================================

def periodic_wrap(pos):
    return pos % L


def minimum_image(dr):
    return dr - L*np.rint(dr/L)


# ============================================================
# FORÇA, ENERGIA E VIRIAL
# ============================================================

def forces_energy_virial(pos, rc):

    dr = (
        pos[:, None, :]
        - pos[None, :, :]
    )

    dr = minimum_image(dr)

    r2 = np.einsum(
        "ijk,ijk->ij",
        dr,
        dr
    )

    mask = (
        (r2 > 0)
        & (r2 < rc**2)
    )

    inv_r2 = np.zeros_like(r2)

    inv_r2[mask] = 1.0/r2[mask]


    # ========================================================
    # FORÇA LJ
    # ========================================================

    scalar = np.zeros_like(r2)

    scalar[mask] = 24 * (
        2*inv_r2[mask]**7
        - inv_r2[mask]**4
    )

    forces = np.einsum(
        "ij,ijk->ik",
        scalar,
        dr
    )


    # ========================================================
    # PARES ÚNICOS
    # ========================================================

    pairs = np.triu(
        mask,
        k=1
    )


    # ========================================================
    # POTENCIAL LJ DESLOCADO
    #
    # U(rc) = 0
    # ========================================================

    U_rc = 4 * (
        rc**(-12)
        - rc**(-6)
    )

    U_pair = np.zeros_like(r2)

    U_pair[mask] = (
        4 * (
            inv_r2[mask]**6
            - inv_r2[mask]**3
        )
        - U_rc
    )

    potential = np.sum(
        U_pair[pairs]
    )


    # ========================================================
    # VIRIAL
    # ========================================================

    W = np.sum(
        (scalar*r2)[pairs]
    )

    return forces, potential, W


# ============================================================
# MEDIDAS
# ============================================================

def measurements():

    K = 0.5*np.sum(v*v)

    T = 2*K/(DIM*N)

    P = (
        N*T
        + virial/DIM
    ) / V

    return (
        K/N,
        U/N,
        (K+U)/N,
        T,
        P
    )


# ============================================================
# LEAPFROG
# ============================================================

def leapfrog():

    global r, v, F, U, virial, time

    dt = dt_slider.val
    rc = rc_slider.val


    # --------------------------------------------------------
    # Meio passo na velocidade
    # --------------------------------------------------------

    v += 0.5*dt*F/mass


    # --------------------------------------------------------
    # Passo completo na posição
    # --------------------------------------------------------

    r += dt*v


    # --------------------------------------------------------
    # Condição periódica
    # --------------------------------------------------------

    r = periodic_wrap(r)


    # --------------------------------------------------------
    # Novas forças
    # --------------------------------------------------------

    F, U, virial = forces_energy_virial(
        r,
        rc
    )


    # --------------------------------------------------------
    # Segundo meio passo na velocidade
    # --------------------------------------------------------

    v += 0.5*dt*F/mass

    time += dt


# ============================================================
# FIGURA
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 9)
)

plt.subplots_adjust(
    left=0.08,
    right=0.97,
    top=0.94,
    bottom=0.26,
    hspace=0.35,
    wspace=0.28
)

ax_sim, ax_E = axes[0]
ax_T, ax_P = axes[1]


# ============================================================
# PAINEL DA SIMULAÇÃO
# ============================================================

particles = ax_sim.scatter(
    [],
    [],
    s=45
)

info = ax_sim.text(
    0.03,
    0.97,
    "",
    transform=ax_sim.transAxes,
    va="top",
    fontsize=9,

    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85
    )
)


# ============================================================
# GRÁFICO DAS ENERGIAS
# ============================================================

line_K, = ax_E.plot(
    [],
    [],
    label=r"$E_K/N$"
)

line_U, = ax_E.plot(
    [],
    [],
    label=r"$E_U/N$"
)

line_E, = ax_E.plot(
    [],
    [],
    linewidth=2,
    label=r"$E/N$"
)

ax_E.set(
    xlabel=r"$t^*$",
    ylabel="Energia reduzida",
    title="Energia"
)

ax_E.grid(alpha=0.25)
ax_E.legend()


# ============================================================
# TEMPERATURA
# ============================================================

line_T, = ax_T.plot(
    [],
    []
)

T0_line = ax_T.axhline(
    T00,
    linestyle="--",
    alpha=0.6,
    label=r"$T_0$"
)

ax_T.set(
    xlabel=r"$t^*$",
    ylabel=r"$T^*$",
    title="Temperatura"
)

ax_T.grid(alpha=0.25)
ax_T.legend()


# ============================================================
# PRESSÃO
# ============================================================

line_P, = ax_P.plot(
    [],
    []
)

ax_P.set(
    xlabel=r"$t^*$",
    ylabel=r"$P^*$",
    title="Pressão pelo virial"
)

ax_P.grid(alpha=0.25)


# ============================================================
# FUNÇÃO AUXILIAR PARA SLIDERS
# ============================================================

def make_slider(
    rect,
    label,
    vmin,
    vmax,
    value,
    step
):

    return Slider(
        fig.add_axes(rect),
        label,
        vmin,
        vmax,
        valinit=value,
        valstep=step
    )


# ============================================================
# SLIDERS
# ============================================================

rho_slider = make_slider(
    [0.10, 0.18, 0.32, 0.025],
    r"$\rho$",
    0.30,
    1.10,
    rho0,
    0.01
)

T_slider = make_slider(
    [0.10, 0.14, 0.32, 0.025],
    r"$T_0$",
    0.20,
    2.00,
    T00,
    0.05
)

dt_slider = make_slider(
    [0.10, 0.10, 0.32, 0.025],
    r"$h=\Delta t$",
    0.001,
    0.010,
    dt0,
    0.001
)

rc_slider = make_slider(
    [0.55, 0.18, 0.25, 0.025],
    r"$r_c$",
    1.20,
    2.50,
    rc0,
    0.05
)

spf_slider = make_slider(
    [0.55, 0.14, 0.25, 0.025],
    "passos/frame",
    1,
    20,
    spf0,
    1
)


# ============================================================
# BOTÕES
# ============================================================

start_button = Button(
    fig.add_axes(
        [0.55, 0.075, 0.11, 0.045]
    ),
    "Iniciar"
)

reset_button = Button(
    fig.add_axes(
        [0.69, 0.075, 0.11, 0.045]
    ),
    "Reset"
)


# ============================================================
# RESET DA SIMULAÇÃO
# ============================================================

def reset_simulation(event=None):

    global running
    global time
    global r, v, F, U, virial


    # ========================================================
    # PARA A SIMULAÇÃO
    # ========================================================

    running = False

    start_button.label.set_text(
        "Iniciar"
    )


    # ========================================================
    # PARÂMETROS
    # ========================================================

    rho = rho_slider.val
    T0 = T_slider.val
    rc = rc_slider.val


    # ========================================================
    # NOVO ESTADO
    # ========================================================

    r = initial_positions(rho)

    v = initial_velocities(T0)

    F, U, virial = forces_energy_virial(
        r,
        rc
    )

    time = 0.0


    # ========================================================
    # LIMPA HISTÓRICO
    # ========================================================

    for hist in (
        t_hist,
        K_hist,
        U_hist,
        E_hist,
        T_hist,
        P_hist
    ):
        hist.clear()


    # ========================================================
    # ATUALIZA CAIXA
    # ========================================================

    ax_sim.set(
        xlim=(0, L),
        ylim=(0, L),
        aspect="equal",
        xlabel=r"$x/\sigma$",
        ylabel=r"$y/\sigma$",
        title="Fluido de Lennard-Jones"
    )

    particles.set_offsets(r)


    # ========================================================
    # LIMPA GRÁFICOS
    # ========================================================

    for line in (
        line_K,
        line_U,
        line_E,
        line_T,
        line_P
    ):
        line.set_data(
            [],
            []
        )

    for ax in (
        ax_E,
        ax_T,
        ax_P
    ):
        ax.set_xlim(
            0,
            10
        )


    # ========================================================
    # ATUALIZA LINHA T0
    # ========================================================

    T0_line.set_ydata(
        [T0, T0]
    )


    update_display()

    fig.canvas.draw_idle()


# ============================================================
# INICIAR / PAUSAR
# ============================================================

def toggle_simulation(event):

    global running

    running = not running

    if running:

        start_button.label.set_text(
            "Pausar"
        )

    else:

        start_button.label.set_text(
            "Continuar"
        )


# ============================================================
# SALVAR MEDIDAS
# ============================================================

def save_data():

    K, Up, E, T, P = measurements()

    t_hist.append(time)

    K_hist.append(K)
    U_hist.append(Up)
    E_hist.append(E)

    T_hist.append(T)
    P_hist.append(P)


    if len(t_hist) > max_history:

        for hist in (
            t_hist,
            K_hist,
            U_hist,
            E_hist,
            T_hist,
            P_hist
        ):
            del hist[:-max_history]

    return K, Up, E, T, P


# ============================================================
# ATUALIZAÇÃO VISUAL
# ============================================================

def update_display():

    K, Up, E, T, P = measurements()


    # ========================================================
    # PARTÍCULAS
    # ========================================================

    particles.set_offsets(r)


    # ========================================================
    # INFORMAÇÕES
    # ========================================================

    info.set_text(

        rf"$N={N}$"
        "\n"
        rf"$\rho={rho_slider.val:.2f}$"
        "\n"
        rf"$L={L:.2f}$"
        "\n"
        rf"$r_c={rc_slider.val:.2f}$"
        "\n"
        rf"$h={dt_slider.val:.4f}$"
        "\n"
        rf"$t={time:.2f}$"
        "\n"
        rf"$T={T:.3f}$"
        "\n"
        rf"$P={P:.3f}$"
        "\n"
        rf"$E/N={E:.4f}$"
    )


# ============================================================
# ATUALIZAÇÃO DOS GRÁFICOS
# ============================================================

def update_graphs():

    line_K.set_data(
        t_hist,
        K_hist
    )

    line_U.set_data(
        t_hist,
        U_hist
    )

    line_E.set_data(
        t_hist,
        E_hist
    )

    line_T.set_data(
        t_hist,
        T_hist
    )

    line_P.set_data(
        t_hist,
        P_hist
    )


    if len(t_hist) < 2:
        return


    # ========================================================
    # JANELA TEMPORAL
    # ========================================================

    xmin = max(
        0,
        time - 10
    )

    xmax = max(
        10,
        time
    )


    for ax in (
        ax_E,
        ax_T,
        ax_P
    ):

        ax.set_xlim(
            xmin,
            xmax
        )

        ax.relim()

        ax.autoscale_view(
            scalex=False,
            scaley=True
        )


# ============================================================
# ANIMAÇÃO
# ============================================================

def animate(frame):

    if not running:
        return


    steps = int(
        spf_slider.val
    )


    # ========================================================
    # INTEGRAÇÃO
    # ========================================================

    for _ in range(steps):

        leapfrog()


    # ========================================================
    # MEDIDAS
    # ========================================================

    save_data()


    # ========================================================
    # GRÁFICOS
    # ========================================================

    update_display()
    update_graphs()


# ============================================================
# CALLBACKS
# ============================================================

start_button.on_clicked(
    toggle_simulation
)

reset_button.on_clicked(
    reset_simulation
)


# Alterar um parâmetro reinicia a experiência
for slider in (
    rho_slider,
    T_slider,
    dt_slider,
    rc_slider
):

    slider.on_changed(
        reset_simulation
    )


# passos/frame pode ser alterado durante a simulação
# sem reiniciar o estado.


# ============================================================
# INICIALIZAÇÃO
# ============================================================

reset_simulation()

ani = FuncAnimation(
    fig,
    animate,
    interval=25,
    cache_frame_data=False
)

plt.show()