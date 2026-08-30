import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button


# ============================================================
# 1. CONSTANTES DO MODELO
# ============================================================

DIM = 2
N = 64

# Unidades reduzidas de Lennard-Jones
sigma = 1.0
epsilon = 1.0
mass = 1.0
kB = 1.0

rho0 = 0.80
T00 = 1.00
dt0 = 0.005
rc0 = 2.50
spf0 = 5

rng = cp.random.default_rng(7)


# ============================================================
# 2. ESTADO GLOBAL
# ============================================================

running = False
time = 0.0

L = 0.0
V = 0.0

r_gpu = None
v_gpu = None
F_gpu = None

U_gpu = cp.array(0.0)
virial_gpu = cp.array(0.0)

# Histórico mantido na CPU para o Matplotlib
t_hist = []
K_hist = []
U_hist = []
E_hist = []
T_hist = []
P_hist = []

max_history = 1500


# ============================================================
# 3. ESTADO INICIAL — GPU
# ============================================================

def initial_positions_gpu(rho):
    global L, V

    L = np.sqrt(N / rho)
    V = L**DIM

    n = int(np.sqrt(N))

    if n*n != N:
        raise ValueError("N deve ser um quadrado perfeito.")

    a = L / n

    grid = (cp.arange(n, dtype=cp.float64) + 0.5) * a

    X, Y = cp.meshgrid(
        grid,
        grid,
        indexing="ij"
    )

    return cp.column_stack(
        (X.ravel(), Y.ravel())
    )


def initial_velocities_gpu(T0):

    theta = rng.uniform(
        0,
        2*cp.pi,
        N
    )

    speed = cp.sqrt(DIM*T0)

    vel = speed * cp.column_stack(
        (
            cp.cos(theta),
            cp.sin(theta)
        )
    )

    # Remove velocidade do centro de massa
    vel -= cp.mean(
        vel,
        axis=0
    )

    # Temperatura inicial
    T = cp.sum(vel*vel) / (DIM*N)

    # Reescala exatamente para T0
    vel *= cp.sqrt(T0/T)

    return vel


# ============================================================
# 4. CONDIÇÕES PERIÓDICAS — GPU
# ============================================================

def periodic_wrap_gpu(pos):
    """Wrap das partículas para dentro da caixa."""
    return cp.mod(pos, L)


def minimum_image_gpu(dr):
    """Convenção da imagem mínima."""
    return dr - L*cp.rint(dr/L)


# ============================================================
# 5. FORÇAS, ENERGIA E VIRIAL — GPU
# ============================================================

def forces_energy_virial_gpu(pos, rc):

    # --------------------------------------------------------
    # r_ij = r_i - r_j
    #
    # Forma: (N, N, DIM)
    # --------------------------------------------------------

    dr = (
        pos[:, None, :]
        - pos[None, :, :]
    )

    dr = minimum_image_gpu(dr)


    # --------------------------------------------------------
    # r_ij²
    # --------------------------------------------------------

    r2 = cp.sum(
        dr*dr,
        axis=2
    )


    # --------------------------------------------------------
    # Máscara do cutoff
    # --------------------------------------------------------

    mask = (
        (r2 > 0.0)
        & (r2 < rc**2)
    )


    # --------------------------------------------------------
    # 1/r²
    # --------------------------------------------------------

    inv_r2 = cp.zeros_like(r2)

    inv_r2[mask] = (
        1.0 / r2[mask]
    )


    # ========================================================
    # FORÇA LJ
    #
    # F_ij =
    #
    # 24 [2 r^-14 - r^-8] r_ij
    #
    # em unidades reduzidas
    # ========================================================

    scalar = cp.zeros_like(r2)

    scalar[mask] = 24.0 * (
        2.0*inv_r2[mask]**7
        - inv_r2[mask]**4
    )


    # --------------------------------------------------------
    # Soma sobre todos os j
    # --------------------------------------------------------

    F = cp.sum(
        scalar[:, :, None] * dr,
        axis=1
    )


    # ========================================================
    # PARES ÚNICOS i < j
    # ========================================================

    pairs = cp.triu(
        mask,
        k=1
    )


    # ========================================================
    # ENERGIA POTENCIAL
    # ========================================================

    U_rc = 4.0 * (
        rc**(-12)
        - rc**(-6)
    )

    U_pair = cp.zeros_like(r2)

    U_pair[mask] = (
        4.0 * (
            inv_r2[mask]**6
            - inv_r2[mask]**3
        )
        - U_rc
    )

    U = cp.sum(
        U_pair[pairs]
    )


    # ========================================================
    # VIRIAL
    #
    # W = Σ r_ij · F_ij
    # ========================================================

    virial = cp.sum(
        (scalar*r2)[pairs]
    )

    return F, U, virial


# ============================================================
# 6. MEDIDAS — GPU
# ============================================================

def measurements_gpu():

    K = 0.5 * cp.sum(
        v_gpu*v_gpu
    )

    T = (
        2.0*K
        / (DIM*N)
    )

    P = (
        N*T
        + virial_gpu/DIM
    ) / V

    return cp.stack([
        K/N,
        U_gpu/N,
        (K + U_gpu)/N,
        T,
        P
    ])


# ============================================================
# 7. LEAPFROG — GPU
# ============================================================

def leapfrog_gpu():

    global r_gpu
    global v_gpu
    global F_gpu
    global U_gpu
    global virial_gpu
    global time

    dt = dt_slider.val
    rc = rc_slider.val


    # --------------------------------------------------------
    # v(t+h/2)
    # --------------------------------------------------------

    v_gpu += (
        0.5 * dt
        * F_gpu / mass
    )


    # --------------------------------------------------------
    # r(t+h)
    # --------------------------------------------------------

    r_gpu += (
        dt * v_gpu
    )


    # --------------------------------------------------------
    # Condições periódicas
    # --------------------------------------------------------

    r_gpu = periodic_wrap_gpu(
        r_gpu
    )


    # --------------------------------------------------------
    # Novas forças
    # --------------------------------------------------------

    F_gpu, U_gpu, virial_gpu = (
        forces_energy_virial_gpu(
            r_gpu,
            rc
        )
    )


    # --------------------------------------------------------
    # Segundo meio passo
    # --------------------------------------------------------

    v_gpu += (
        0.5 * dt
        * F_gpu / mass
    )

    time += dt


# ============================================================
# 8. FIGURA
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
# 9. PAINEL DO FLUIDO
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
# 10. ENERGIAS
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
    lw=2,
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
# 11. TEMPERATURA
# ============================================================

line_T, = ax_T.plot(
    [],
    []
)

T0_line = ax_T.axhline(
    T00,
    ls="--",
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
# 12. PRESSÃO
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
# 13. SLIDERS
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
    50,
    spf0,
    1
)


# ============================================================
# 14. BOTÕES
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
# 15. RESET
# ============================================================

def reset_simulation(event=None):

    global running
    global time

    global r_gpu
    global v_gpu
    global F_gpu
    global U_gpu
    global virial_gpu


    running = False
    time = 0.0

    start_button.label.set_text(
        "Iniciar"
    )

    rho = rho_slider.val
    T0 = T_slider.val
    rc = rc_slider.val


    # ========================================================
    # NOVO ESTADO NA GPU
    # ========================================================

    r_gpu = initial_positions_gpu(
        rho
    )

    v_gpu = initial_velocities_gpu(
        T0
    )

    F_gpu, U_gpu, virial_gpu = (
        forces_energy_virial_gpu(
            r_gpu,
            rc
        )
    )


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
    # GPU -> CPU
    # apenas para exibição
    # ========================================================

    r_cpu = cp.asnumpy(
        r_gpu
    )

    particles.set_offsets(
        r_cpu
    )


    # ========================================================
    # CAIXA
    # ========================================================

    ax_sim.set(
        xlim=(0, L),
        ylim=(0, L),
        aspect="equal",
        xlabel=r"$x/\sigma$",
        ylabel=r"$y/\sigma$",
        title="Fluido de Lennard-Jones"
    )


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
        line.set_data([], [])


    for ax in (
        ax_E,
        ax_T,
        ax_P
    ):
        ax.set_xlim(
            0,
            10
        )


    T0_line.set_ydata(
        [T0, T0]
    )

    update_display()

    fig.canvas.draw_idle()


# ============================================================
# 16. INICIAR / PAUSAR
# ============================================================

def toggle_simulation(event):

    global running

    running = not running

    start_button.label.set_text(
        "Pausar"
        if running
        else "Continuar"
    )


# ============================================================
# 17. TRANSFERÊNCIA GPU -> CPU
# ============================================================

def get_frame_data():
    """
    Realiza apenas uma transferência principal de dados
    da GPU para CPU por frame.
    """

    M_gpu = measurements_gpu()

    # Junta posições e propriedades
    data_gpu = cp.concatenate([
        r_gpu.ravel(),
        M_gpu
    ])

    data_cpu = cp.asnumpy(
        data_gpu
    )

    positions = data_cpu[:N*DIM].reshape(
        N,
        DIM
    )

    K, Up, E, T, P = data_cpu[N*DIM:]

    return (
        positions,
        K,
        Up,
        E,
        T,
        P
    )


# ============================================================
# 18. HISTÓRICO
# ============================================================

def save_data(K, Up, E, T, P):

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


# ============================================================
# 19. DISPLAY
# ============================================================

def update_display():

    (
        positions,
        K,
        Up,
        E,
        T,
        P
    ) = get_frame_data()


    particles.set_offsets(
        positions
    )


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

    return K, Up, E, T, P


# ============================================================
# 20. GRÁFICOS
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
# 21. ANIMAÇÃO
# ============================================================

def animate(frame):

    if not running:
        return


    # ========================================================
    # VÁRIOS PASSOS TOTALMENTE NA GPU
    # ========================================================

    nsteps = int(
        spf_slider.val
    )

    for _ in range(nsteps):

        leapfrog_gpu()


    # ========================================================
    # UMA TRANSFERÊNCIA GPU -> CPU
    # ========================================================

    K, Up, E, T, P = (
        update_display()
    )


    # ========================================================
    # HISTÓRICO
    # ========================================================

    save_data(
        K,
        Up,
        E,
        T,
        P
    )

    update_graphs()


# ============================================================
# 22. CALLBACKS
# ============================================================

start_button.on_clicked(
    toggle_simulation
)

reset_button.on_clicked(
    reset_simulation
)


for slider in (
    rho_slider,
    T_slider,
    dt_slider,
    rc_slider
):

    slider.on_changed(
        reset_simulation
    )


# passos/frame apenas muda a velocidade de execução


# ============================================================
# 23. INICIALIZAÇÃO
# ============================================================

reset_simulation()

ani = FuncAnimation(
    fig,
    animate,
    interval=25,
    cache_frame_data=False
)

plt.show()