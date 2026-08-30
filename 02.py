import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.patches import Circle, FancyArrowPatch

# ===== KERNEL CUDA (Velocity Verlet) =====
verlet_kernel = cp.RawKernel(r'''
extern "C" __global__ void verlet(double *x, double *v, double *time,
    double sigma, double epsilon, double mass, double dt, int nsteps, int wca) {
    if (threadIdx.x || blockIdx.x) return;
    const double rc = 1.122462048309373 * sigma;
    for (int n=0; n<nsteps; n++) {
        double rij = x[0]-x[1], r = fabs(rij);
        if (r<1e-8) r=1e-8;
        double sr = sigma/r;
        double F = (48.0*epsilon/(sigma*sigma))*(pow(sr,14.)-0.5*pow(sr,8.))*rij;
        if (wca && r>=rc) F=0.0;
        double a1=F/mass, a2=-F/mass;
        double x1n=x[0]+v[0]*dt+0.5*a1*dt*dt;
        double x2n=x[1]+v[1]*dt+0.5*a2*dt*dt;
        rij=x1n-x2n; r=fabs(rij);
        if (r<1e-8) r=1e-8;
        sr=sigma/r;
        double Fn=(48.0*epsilon/(sigma*sigma))*(pow(sr,14.)-0.5*pow(sr,8.))*rij;
        if (wca && r>=rc) Fn=0.0;
        double a1n=Fn/mass, a2n=-Fn/mass;
        v[0]+=0.5*(a1+a1n)*dt; v[1]+=0.5*(a2+a2n)*dt;
        x[0]=x1n; x[1]=x2n; time[0]+=dt;
    }
}
''', 'verlet')

# ===== FUNÇÕES GPU =====
def cutoff(s): return 2**(1/6)*s
def radial_force_gpu(r, s, e, model):
    r = cp.asarray(r, dtype=cp.float64)
    F = (48*e/s)*((s/r)**13 - 0.5*(s/r)**7)
    if model=="WCA": F = cp.where(r < cutoff(s), F, 0.0)
    return F

def instant_force_gpu(x, s, e, model):
    rij = x[0]-x[1]; r = cp.maximum(cp.abs(rij), 1e-8)
    F = (48*e/s**2)*((s/r)**14 - 0.5*(s/r)**8)*rij
    if model=="WCA": F = cp.where(r < cutoff(s), F, 0.0)
    return F

# ===== PARÂMETROS E ESTADO (GPU) =====
s0=e0=1.0; mass=1.0; v0=0.45; dt=1e-3; substeps=10
x_gpu = cp.array([-1.5,1.5], dtype=cp.float64)
v_gpu = cp.array([v0,-v0], dtype=cp.float64)
time_gpu = cp.array([0.0], dtype=cp.float64)
running = False
t_hist, x1_hist, x2_hist = [], [], []

# ===== FIGURA =====
fig = plt.figure(figsize=(13,8))
plt.subplots_adjust(left=.07, right=.95, top=.93, bottom=.22, hspace=.42, wspace=.30)
gs = fig.add_gridspec(2,2, height_ratios=[1,1])
ax_atoms, ax_force, ax_pos = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,:])

# ===== ÁTOMOS E ELEMENTOS =====
ax_atoms.set(xlim=(-4,4), ylim=(-1.5,1.5), aspect="equal", yticks=[], xlabel="Posição $x$", title="Dinâmica")
ax_atoms.axhline(0, c='gray', lw=.8, alpha=.5)
atoms = [Circle((-1.5,0), s0/2, fc='silver', ec='k', lw=1.5) for _ in range(2)]
for a in atoms: ax_atoms.add_patch(a)
labels = [ax_atoms.text(x,0,f"${q}$", ha='center', va='center', fontsize=13, weight='bold')
          for x,q in zip((-1.5,1.5),('i','j'))]
arrows = [FancyArrowPatch((0,.65),(0,.65), arrowstyle='-|>', mutation_scale=15, lw=2) for _ in range(2)]
for a in arrows: ax_atoms.add_patch(a)
info = ax_atoms.text(.02,.96,"", transform=ax_atoms.transAxes, va='top', fontsize=10,
                     bbox=dict(boxstyle='round', fc='white', alpha=.85))

# ===== GRÁFICO F(r) =====
r_curve = cp.linspace(.65,4,3000)
force_line, = ax_force.plot([],[],'k',lw=2)
force_point, = ax_force.plot([],[],'o',ms=8,zorder=10)
rij_line = ax_force.axvline(0, ls='--', lw=1.2)
ax_force.axhline(0, c='k', lw=.8)
ax_force.set(xlim=(.65,4), ylim=(-5,60), xlabel=r"$|r_{ij}|$", ylabel=r"$F(r)$", title="Força instantânea")
ax_force.grid(alpha=.25)

# ===== TRAJETÓRIAS =====
lines = [ax_pos.plot([],[], lw=1.8, label=lab)[0] for lab in (r"$x_i(t)$", r"$x_j(t)$")]
ax_pos.set(xlim=(0,10), ylim=(-4,4), xlabel="Tempo $t$", ylabel="Posição $x(t)$", title="Trajetórias")
ax_pos.grid(alpha=.25); ax_pos.legend()

# ===== CONTROLES =====
def slider(rect,label,a,b,init,step):
    return Slider(fig.add_axes(rect), label, a, b, valinit=init, valstep=step)
sld_s = slider([.10,.135,.40,.025], r"$\sigma$", .6,1.5, s0,.02)
sld_e = slider([.10,.095,.40,.025], r"$\epsilon$", .3,2.0, e0,.02)
sld_v = slider([.10,.055,.40,.025], r"$v_0$", .05,1.0, v0,.01)
model_sel = RadioButtons(fig.add_axes([.57,.075,.12,.10]), ("LJ","WCA"))
start_btn = Button(fig.add_axes([.73,.115,.09,.045]), "Iniciar")
reset_btn = Button(fig.add_axes([.84,.115,.09,.045]), "Reiniciar")

# ===== FUNÇÕES DE ATUALIZAÇÃO =====
def pars(): return sld_s.val, sld_e.val, model_sel.value_selected

def integrate_gpu():
    s,e,model = pars()
    verlet_kernel((1,),(1,), (x_gpu, v_gpu, time_gpu, np.float64(s), np.float64(e),
                 np.float64(mass), np.float64(dt), np.int32(substeps), np.int32(model=="WCA")))

regions = []
def update_force_curve():
    s,e,model = pars()
    F = cp.asnumpy(radial_force_gpu(r_curve, s, e, model))
    force_line.set_data(cp.asnumpy(r_curve), F)
    for r in regions: r.remove()
    regions.clear()
    rc = cutoff(s)
    regions.append(ax_force.axvspan(.65, rc, color='red', alpha=.10))
    if model=="LJ": regions.append(ax_force.axvspan(rc,4, color='blue', alpha=.08))

def update_visuals():
    s,e,model = pars()
    F = instant_force_gpu(x_gpu, s, e, model)
    rij = cp.abs(x_gpu[0]-x_gpu[1])
    Fr = radial_force_gpu(rij, s, e, model)
    data = cp.array([x_gpu[0], x_gpu[1], v_gpu[0], v_gpu[1], rij, F, Fr, time_gpu[0]])
    x1,x2,v1,v2,rij,F,Fr,t = cp.asnumpy(data)
    rc = cutoff(s)
    color = 'gray' if (model=="WCA" and rij>=rc) else ('red' if rij<rc else 'blue')
    for k,x in enumerate((x1,x2)):
        atoms[k].center=(x,0); atoms[k].radius=s/2; labels[k].set_position((x,0))
    length = min(abs(F)*.035, 1.2)
    for k,(x,f) in enumerate(zip((x1,x2),(F,-F))):
        end = x if abs(f)<1e-12 else x + np.sign(f)*length
        arrows[k].set_positions((x,.70),(end,.70)); arrows[k].set_color(color)
    info.set_text(f"$t={t:.3f}$\n$|r_{{ij}}|={rij:.3f}$\n$F_{{ij}}={F:.3f}$\n$v_i={v1:.3f}$\n$v_j={v2:.3f}$")
    force_point.set_data([rij],[Fr]); force_point.set_color(color)
    rij_line.set_xdata([rij,rij])
    return t, x1, x2

def history(t,x1,x2):
    t_hist.append(t); x1_hist.append(x1); x2_hist.append(x2)
    if len(t_hist)>4000:
        for lst in (t_hist,x1_hist,x2_hist): del lst[:-4000]
    lines[0].set_data(t_hist,x1_hist); lines[1].set_data(t_hist,x2_hist)
    ax_pos.set_xlim(max(0,t-10), max(10,t))

# ===== ANIMAÇÃO =====
def animate(_):
    if running:
        integrate_gpu()
        t,x1,x2 = update_visuals()
        history(t,x1,x2)
    else:
        update_visuals()

# ===== CALLBACKS =====
def toggle(_):
    global running
    running = not running
    start_btn.label.set_text("Pausar" if running else "Continuar")

def reset(_=None):
    global running
    running = False
    start_btn.label.set_text("Iniciar")
    x_gpu[:] = cp.array([-1.5,1.5])
    v = sld_v.val
    v_gpu[:] = cp.array([v,-v])
    time_gpu[0] = 0.0
    t_hist.clear(); x1_hist.clear(); x2_hist.clear()
    for line in lines: line.set_data([],[])
    ax_pos.set_xlim(0,10)
    update_force_curve()
    update_visuals()
    fig.canvas.draw_idle()

start_btn.on_clicked(toggle)
reset_btn.on_clicked(reset)
for s in (sld_s,sld_e,sld_v): s.on_changed(reset)
model_sel.on_clicked(reset)

# ===== INICIALIZAÇÃO =====
update_force_curve()
update_visuals()
ani = FuncAnimation(fig, animate, interval=20, cache_frame_data=False)
plt.show()