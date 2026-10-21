import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import h, hbar
from itertools import combinations


### Numerical constants for Rubidium 87 (D2 transition)

Gamma_0 = 2 * np.pi * 6.0666e6  # rad/s

lambda_0 = 780.241209686e-9  # m
k_0 = 2 * np.pi / lambda_0

m = 1.443160895e-25  # kg


### Optical tweezers parameters

U_0 = h * 20e6  # J
w_0 = 700e-9  # m
lambda_L = 850e-9  # m

omega_r = np.sqrt(4 * U_0 / (m * w_0**2))
omega_z = omega_r * lambda_L / (np.sqrt(2) * np.pi * w_0)


### Atomic geometry

N = 2
geometry = "circle"

# d = distance between nearest-neighbor atoms
d_list = np.linspace(20e-9, 200e-9, 200)
d_plot = 50e-9


def get_positions(d, geometry, N):

    if geometry == "circle":

        theta = 2 * np.pi * np.arange(N) / N

        # d is the nearest-neighbor distance, not the radius.
        R = d / (2 * np.sin(np.pi / N))

        x = R * np.cos(theta)
        y = R * np.sin(theta)
        z = np.zeros(N)

        return np.column_stack((x, y, z))

    raise ValueError(f"Unknown geometry: {geometry}")

positions = get_positions(d_plot, geometry, N)

x_atoms = positions[:, 0] * 1e9
y_atoms = positions[:, 1] * 1e9

fig, ax = plt.subplots(figsize=(6,6))

ax.scatter(
    x_atoms, y_atoms,
    s=150,
    color="tab:blue",
    edgecolors="black",
    zorder=3
)

for i in range(N):
    ax.annotate(
        f"{i+1}",
        (x_atoms[i], y_atoms[i]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=12
    )

# Link the atoms
ax.plot(
    np.append(x_atoms, x_atoms[0]),
    np.append(y_atoms, y_atoms[0]),
    "--",
    color="gray",
    linewidth=1.2,
    zorder=1
)

ax.set_xlabel("x (nm)", fontsize=14)
ax.set_ylabel("y (nm)", fontsize=14)

# Limits
margin = 0.3 * d_plot * 1e9
xmin, xmax = x_atoms.min(), x_atoms.max()
ymin, ymax = y_atoms.min(), y_atoms.max()

ax.set_xlim(xmin - margin, xmax + margin)
ax.set_ylim(ymin - margin, ymax + margin)

ax.set_aspect("equal", adjustable="box")
ax.grid(True)

plt.tight_layout()

# plt.show()


### Atomic operators

# Basis convention: |e> = [1, 0], |g> = [0, 1]
sigma = np.array([
    [0, 0],
    [1, 0]
], dtype=complex)

I_2 = np.eye(2, dtype=complex)

# Hilbert space dimension
D = 2**N

# Lowering operators sigma_i acting on atom i
sigma_list = []

for i in range(N):

    operators = [
        sigma if j == i else I_2
        for j in range(N)
    ]

    sigma_i = operators[0]

    for op in operators[1:]:
        sigma_i = np.kron(sigma_i, op)

    sigma_list.append(sigma_i)

# Excited-state number operator
N_e_operator = sum(
    s.conj().T @ s
    for s in sigma_list
)


### Pairwise coherent and dissipative interactions

def get_interactions(d, geometry, N):

    positions = get_positions(d, geometry, N)

    # Distance matrix
    distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :],
        axis=2
    )

    # Dimensionless separation k_0 r_ij
    x = k_0 * distances

    # Interaction matrices
    J = np.zeros((N, N))
    Gamma = np.zeros((N, N))

    for i in range(N):

        # Single-atom decay
        Gamma[i, i] = Gamma_0

        for j in range(i + 1, N):

            x_ij = x[i, j]

            # Coherent interaction
            J_ij = (-3 / 4) * Gamma_0 * (
                np.cos(x_ij) / x_ij
                - np.sin(x_ij) / x_ij**2
                - np.cos(x_ij) / x_ij**3
            )

            # Collective decay
            Gamma_ij = (3 / 2) * Gamma_0 * (
                np.sin(x_ij) / x_ij
                + np.cos(x_ij) / x_ij**2
                - np.sin(x_ij) / x_ij**3
            )

            J[i, j] = J_ij
            J[j, i] = J_ij

            Gamma[i, j] = Gamma_ij
            Gamma[j, i] = Gamma_ij

    return positions, J, Gamma


### Hamiltonian

def get_hamiltonian(J):

    H = np.zeros((D, D), dtype=complex)

    for i, j in combinations(range(N), 2):

        H += hbar * J[i, j] * (
            sigma_list[i].conj().T @ sigma_list[j]
            + sigma_list[j].conj().T @ sigma_list[i]
        )

    return H


### Lindblad master equations

def make_lindblad_collective(H, Gamma):

    def lindblad_collective(t, rho_vec):

        rho = rho_vec.reshape((D, D))

        # Coherent evolution
        drho = -1j / hbar * (
            H @ rho - rho @ H
        )

        # Collective dissipation:
        # sum_ij Gamma_ij D[sigma_i, sigma_j](rho)
        for i in range(N):
            for j in range(N):

                si = sigma_list[i]
                sj = sigma_list[j]

                drho += Gamma[i, j] * (
                    si @ rho @ sj.conj().T
                    - 0.5 * (
                        sj.conj().T @ si @ rho
                        + rho @ sj.conj().T @ si
                    )
                )

        return drho.flatten()

    return lindblad_collective


def lindblad_independent(t, rho_vec):

    rho = rho_vec.reshape((D, D))

    drho = np.zeros((D, D), dtype=complex)

    for si in sigma_list:

        drho += Gamma_0 * (
            si @ rho @ si.conj().T
            - 0.5 * (
                si.conj().T @ si @ rho
                + rho @ si.conj().T @ si
            )
        )

    return drho.flatten()


### Initial state: all atoms excited

e = np.array([1, 0], dtype=complex)

psi_0 = e

for _ in range(N - 1):
    psi_0 = np.kron(psi_0, e)

rho_0 = np.outer(psi_0, psi_0.conj())


### Time evolution for a chosen nearest-neighbor distance

positions, J, Gamma = get_interactions(
    d_plot, geometry, N
)

H = get_hamiltonian(J)

lindblad_collective = make_lindblad_collective(H, Gamma)

# Dimensionless time Gamma_0 t
t_adim = np.linspace(0, 6, 200)
t = t_adim / Gamma_0

# Independent atoms
solution_independent = solve_ivp(
    lindblad_independent,
    (t[0], t[-1]),
    rho_0.flatten(),
    t_eval=t,
    rtol=1e-7,
    atol=1e-9
)

rho_independent = solution_independent.y.T.reshape(
    -1, D, D
)

# Collective emission
solution_collective = solve_ivp(
    lindblad_collective,
    (t[0], t[-1]),
    rho_0.flatten(),
    t_eval=t,
    rtol=1e-7,
    atol=1e-9
)

rho_collective = solution_collective.y.T.reshape(
    -1, D, D
)


### Excited correlation matrix

# Excitation number operators
n_list = [
    sigma_list[i].conj().T @ sigma_list[i]
    for i in range(N)
]

# Product
nn_list = [
    [
        n_list[i] @ n_list[j]
        for j in range(N)
    ]
    for i in range(N)
]


def get_correlations(rho_t):
    """
    Calcule C_ij(t) pour toutes les matrices de densité rho_t.

    Paramètre
    ---------
    rho_t : array de forme (n_times, D, D)

    Retour
    ------
    C : array de forme (n_times, N, N)
    """
    n_times = len(rho_t)
    C = np.zeros((n_times, N, N), dtype=float)

    for k, rho in enumerate(rho_t):

        # <n_i> pour chaque atome
        mean_n = np.array([
            np.real(np.trace(n_list[i] @ rho))
            for i in range(N)
        ])

        for i in range(N):
            for j in range(N):

                # <n_i n_j>
                mean_nn = np.real(
                    np.trace(nn_list[i][j] @ rho)
                )

                C[k, i, j] = 4 * (
                    mean_nn - mean_n[i] * mean_n[j]
                )

    return C


C_independent = get_correlations(rho_independent)
C_collective = get_correlations(rho_collective)


### Plot

tau_plot = 1
idx = np.argmin(np.abs(t_adim - tau_plot))

C1 = C_independent[idx]
C2 = C_collective[idx]

# Common color scale
vmax = max(
    np.max(np.abs(C1)),
    np.max(np.abs(C2))
)
if vmax == 0:
    vmax = 1e-12

vmin = -vmax

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, C, title in zip(
    axes,
    [C1, C2],
    ["Independent atoms", "Collective emission"]
):
    im = ax.imshow(
        C,
        origin="upper",  # Row 0 at the top
        cmap="RdBu_r",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax
    )

    # Numerical values inside each cell
    for i in range(N):
        for j in range(N):
            value = C[i, j]

            text_color = (
                "white" if abs(value) > 0.5 * vmax else "black"
            )

            ax.text(
                j, i,  # x = column j, y = row i
                f"{value:.4f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="bold"
            )

    ax.set_xlabel(r"Atom $j$")
    ax.set_ylabel(r"Atom $i$")
    ax.set_title(title, fontsize=12)

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(range(1, N + 1))
    ax.set_yticklabels(range(1, N + 1))

    fig.colorbar(
        im,
        ax=ax,
        label=r"$C_{ij}$",
        fraction=0.035,
        pad=0.02,
        shrink=0.75
    )

fig.suptitle(
    rf"Excitation correlation matrix at "
    rf"$\Gamma_0 t={t_adim[idx]:.2f}$",
    fontsize=14
)

plt.tight_layout()
plt.show()