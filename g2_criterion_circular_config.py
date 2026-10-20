import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, hbar, c

### Physical constants

# Rb 87
m = 1.443160895e-25  # kg

# D2 transition
Gamma_0 = 2 * np.pi * 6.0666e6  # rad/s
lambda_0 = 780.241209686e-9  # m
k_0 = 2 * np.pi / lambda_0

# Optical tweezers
U_0 = h * 20e6  # J
w_0 = 700e-9  # m
lambda_L = 850e-9  # m

omega_r = np.sqrt(4 * U_0 / (m * w_0**2))
omega_z = omega_r * lambda_L / (np.sqrt(2) * np.pi * w_0)

# Position uncertainty
Delta_z = np.sqrt(hbar / (2 * m * omega_z))

print(f"omega_z / (2*pi) = {omega_z/(2*np.pi)*1e-3:.3e} kHz")
print(f"Delta_z = {Delta_z*1e9:.3f} nm")


### Atomic geometry

N = 5
geometry = "circle"

d_list = np.linspace(20e-9, 200e-9, 200)
d_plot = 100e-9

theta = 2 * np.pi * np.arange(N) / N


def get_positions(d, geometry, N):

    if geometry == "circle":
        theta = 2 * np.pi * np.arange(N) / N

        x = d * np.cos(theta)
        y = d * np.sin(theta)
        z = np.zeros(N)

        return np.column_stack((x, y, z))

    raise ValueError(f"Unknown geometry : {geometry}")


### Green tensor and Interaction matrix

def G(positions, k_0):

    positions = np.asarray(positions, dtype=float)

    r_ij = positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(r_ij, axis=-1)

    r_outer = np.einsum("ijk,ijl->ijkl", r_ij, r_ij)

    Id = np.eye(3)

    # Avoid divison by zero
    r_safe = np.where(r == 0, 1.0, r)

    x = k_0 * r_safe

    A = x**2 + 1j*x - 1
    B = -x**2 - 3j*x + 3

    green = (
        np.exp(1j*x) / (4*np.pi*k_0**2*r_safe**3)
    )[..., None, None] * (
        A[..., None, None] * Id
        + B[..., None, None]
        * r_outer / r_safe[..., None, None]**2
    )

    return green


def Gamma_scalar_matrix(positions, Gamma_0, k_0):

    green = G(positions, k_0)

    # Projection onto the polarization e_z
    G_zz = green[:, :, 2, 2]

    Gamma = (6*np.pi*Gamma_0/k_0) * np.imag(G_zz)

    # Diagonal
    np.fill_diagonal(Gamma, Gamma_0)

    return Gamma


### g2

def g2_from_positions(positions):

    Gamma = Gamma_scalar_matrix(positions, Gamma_0, k_0)

    # Symmetric real matrix
    eigenvalues = np.linalg.eigvalsh(Gamma)

    variance = np.var(eigenvalues / Gamma_0)

    return 1 + (variance - 1)/N


### Monte Carlo

N_MC = 1000

g2_MC = np.zeros((N_MC, len(d_list)))

for mc in range(N_MC):

    # One z configuration per position
    z_list = np.random.normal(0, Delta_z, size=N)

    for j, d in enumerate(d_list):

        # xy positions
        positions = get_positions(d, geometry, N)

        # Fluctuations
        positions[:, 2] = z_list

        g2_MC[mc, j] = g2_from_positions(positions)


### Ensemble statistics

g2_mean = np.mean(g2_MC, axis=0)

g2_p5 = np.percentile(g2_MC, 5, axis=0)
g2_p95 = np.percentile(g2_MC, 95, axis=0)

d_nm = d_list * 1e9


### g2 plot

plt.figure(figsize=(9, 5))

# 5–95 % band
plt.fill_between(
    d_nm,
    g2_p5,
    g2_p95,
    color="tab:blue",
    alpha=0.25,
    label="Percentiles 5–95 %"
)

# Mean
plt.plot(
    d_nm,
    g2_mean,
    color="tab:blue",
    linewidth=2.5,
    label=r"Mean $\langle g^{(2)}(0)\rangle$"
)

plt.axhline(
    1,
    color="red",
    linestyle=":",
    linewidth=2,
    label=r"$g^{(2)}(0)=1$"
)

plt.xlabel(r"$d$ (nm)")
plt.ylabel(r"$g^{(2)}(0)$")

plt.title(
    rf'Monte Carlo over the positions $z_i$ : '
    rf'$N={N}$, $\Delta z={Delta_z*1e9:.2f}$ nm, '
    rf'$N_{{MC}}={N_MC}$'
)

plt.grid(True)
plt.legend()
plt.tight_layout()


### Atomic configuration plot

positions_plot = get_positions(d_plot, geometry, N)

plt.figure(figsize=(5, 5))

plt.scatter(
    positions_plot[:, 0]*1e9,
    positions_plot[:, 1]*1e9,
    s=100,
    label="Atoms"
)

phi = np.linspace(0, 2*np.pi, 300)

plt.plot(
    d_plot*np.cos(phi)*1e9,
    d_plot*np.sin(phi)*1e9,
    linestyle=":",
    linewidth=1.5
)

plt.xlabel(r"$x$ (nm)")
plt.ylabel(r"$y$ (nm)")
plt.title(rf"Circular configuration, $N={N}$, $d={d_plot*1e9:.0f}$ nm")

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()