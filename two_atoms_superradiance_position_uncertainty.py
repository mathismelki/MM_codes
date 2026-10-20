import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import h, hbar

### Numerical constants for Rubidium 87 (D2 transition)

# Natural line width
Gamma_0 = 2 * np.pi * 6.0666e6  # rad/s

# Wavelength and wave vector
lambda_0 = 780.241209686e-9  # m
k_0 = 2 * np.pi / lambda_0

# Mass
m = 1.443160895e-25 # kg

### Optical tweezers parameters

U_0 = h * 20e6 # Hz
w_0 = 700e-9 # m
lambda_L = 850e-9 # m

omega_r = np.sqrt((4*U_0) / (m * w_0**2)) # rad/s
omega_z = omega_r * (lambda_L)/(np.sqrt(2) * np.pi * w_0) # rad/s

### Parameters

# r distance
r = 100e-9 # m

# z positions
z1, z2 = 0, 0

# Single-atom lowering operator
sigma = np.array([
    [0, 0],
    [1, 0]
], dtype=complex)

# Identity operator
I = np.eye(2, dtype=complex)

# Lowering operators for atoms 1 and 2
sigma_1 = np.kron(sigma, I)
sigma_2 = np.kron(I, sigma)

### Interacting atoms

z = z2 - z1

# Distance between the atoms
d = np.sqrt(r**2 + z**2)
x = k_0 * d
q = z**2 / d**2

# Coherent interaction rate
J_12 = (-3/4) * Gamma_0 * (
    (1 - q) * np.cos(x) / x
    - (1 - 3*q) * np.sin(x) / x**2
    - (1 - 3*q) * np.cos(x) / x**3
)

# Dissipative interaction rate
Gamma_12 = (3/2) * Gamma_0 * (
    (1 - q) * np.sin(x) / x
    + (1 - 3*q) * np.cos(x) / x**2
    - (1 - 3*q) * np.sin(x) / x**3
)

# Interaction Hamiltonian
H_int = hbar * J_12 * (
    sigma_1.conj().T @ sigma_2
    + sigma_2.conj().T @ sigma_1
)

H = H_int # in the rotating frame

# Collective decay rates
Gamma_plus = Gamma_0 + Gamma_12
Gamma_minus = Gamma_0 - Gamma_12

# Collective jump operators
O_plus = (sigma_1 + sigma_2) / np.sqrt(2)
O_minus = (sigma_1 - sigma_2) / np.sqrt(2)

### Lindblad master equations

# Collective case
def lindblad_collective(t, rho_vec):
    rho = rho_vec.reshape((4, 4))

    drho = (
        -1j / hbar * (
            H @ rho - rho @ H
        )
        + Gamma_plus * (
            O_plus @ rho @ O_plus.conj().T
            - 0.5 * (
                O_plus.conj().T @ O_plus @ rho
                + rho @ O_plus.conj().T @ O_plus
            )
        )
        + Gamma_minus * (
            O_minus @ rho @ O_minus.conj().T
            - 0.5 * (
                O_minus.conj().T @ O_minus @ rho
                + rho @ O_minus.conj().T @ O_minus
            )
        )
    )

    return drho.flatten()

# Independent atoms
def lindblad_independent(t, rho_vec):
    rho = rho_vec.reshape((4, 4))

    drho = np.zeros((4, 4), dtype=complex)

    for s in (sigma_1, sigma_2):
        drho += Gamma_0 * (
            s @ rho @ s.conj().T
            - 0.5 * (
                s.conj().T @ s @ rho
                + rho @ s.conj().T @ s
            )
        )

    return drho.flatten()

### Initial state

e = np.array([1, 0], dtype=complex)
psi_0 = np.kron(e, e)
rho_0 = np.outer(psi_0, psi_0.conj())

### Monte Carlo: uncertainty in inter-atomic distance

# Position uncertainty
Delta_z1, Delta_z2 = np.sqrt(hbar / (2*m*omega_z)), np.sqrt(hbar / (2*m*omega_z)) # m

# Number of Monte Carlo realizations
N = 5000

z1_min, z1_max = z1 - Delta_z1, z1 + Delta_z1
z2_min, z2_max = z2 - Delta_z2, z2 + Delta_z2

# Initial random samples
z1_samples = np.random.normal(z1, Delta_z1, N)
z2_samples = np.random.normal(z2, Delta_z2, N)

z_samples = z2_samples - z1_samples

d_samples = (r**2 + z_samples**2)**(1/2)
print("Number of samples:", len(d_samples))
print("Minimum distance (nm):", np.min(d_samples) * 1e9)
print("Maximum distance (nm):", np.max(d_samples) * 1e9)

plt.figure(figsize=(8, 5))

plt.hist(
    d_samples * 1e9,
    bins=50,
    density=True,
    alpha=0.55,
    color="tab:blue",
    edgecolor="white",
    label=rf"Monte Carlo ($d_i>0$, $N={N}$)"
)

plt.xlabel("Interatomic distance (nm)", fontsize=13)
plt.ylabel("Probability density (nm$^{-1}$)", fontsize=13)
plt.title("Interatomic distances distribution", fontsize=14)
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()

#plt.show() # Figure 1

###

# Dimensionless time
t_adim = np.linspace(0, 3, 200)
t = t_adim / Gamma_0

# Independent-atom reference
solution_independent = solve_ivp(
    lindblad_independent,
    (t[0], t[-1]),
    rho_0.flatten(),
    t_eval=t
)

rho_independent = solution_independent.y.T.reshape(-1, 4, 4)

R_independent = np.array([
    Gamma_0 * np.real(
        np.trace(sigma_1.conj().T @ sigma_1 @ rho)
        + np.trace(sigma_2.conj().T @ sigma_2 @ rho)
    )
    for rho in rho_independent
])

def simulate_collective(z1_sample, z2_sample):

    z_sample = z2_sample - z1_sample
    d_sample = np.sqrt(r**2 + z_sample**2)
    x_sample = k_0 * d_sample
    q_sample = z_sample**2 / d_sample**2

    J = (-3/4) * Gamma_0 * (
        (1 - q_sample) * np.cos(x_sample) / x_sample
        - (1 - 3*q_sample) * np.sin(x_sample) / x_sample**2
        - (1 - 3*q_sample) * np.cos(x_sample) / x_sample**3
    )

    Gamma_12_sample = (3/2) * Gamma_0 * (
        (1 - q_sample) * np.sin(x_sample) / x_sample
        + (1 - 3*q_sample) * np.cos(x_sample) / x_sample**2
        - (1 - 3*q_sample) * np.sin(x_sample) / x_sample**3
    )

    Gamma_p = Gamma_0 + Gamma_12_sample
    Gamma_m = Gamma_0 - Gamma_12_sample

    H_sample = hbar * J * (
        sigma_1.conj().T @ sigma_2
        + sigma_2.conj().T @ sigma_1
    )

    def rhs(t_now, rho_vec):
        rho = rho_vec.reshape(4, 4)

        drho = -1j / hbar * (
            H_sample @ rho - rho @ H_sample
        )

        for rate, O in (
            (Gamma_p, O_plus),
            (Gamma_m, O_minus)
        ):
            OdagO = O.conj().T @ O
            drho += rate * (
                O @ rho @ O.conj().T
                - 0.5 * (OdagO @ rho + rho @ OdagO)
            )

        return drho.flatten()

    sol = solve_ivp(
        rhs,
        (t[0], t[-1]),
        rho_0.flatten(),
        t_eval=t,
        rtol=1e-8,
        atol=1e-10
    )

    rho_t = sol.y.T.reshape(-1, 4, 4)

    R = np.array([
        np.real(
            Gamma_p * np.trace(O_plus.conj().T @ O_plus @ rho)
            + Gamma_m * np.trace(O_minus.conj().T @ O_minus @ rho)
        )
        for rho in rho_t
    ])

    return R


# Run Monte Carlo
R_MC = np.array([
    simulate_collective(z1_sample, z2_sample)
    for z1_sample, z2_sample in zip(z1_samples, z2_samples)
])

# Ensemble statistics
R_mean = np.mean(R_MC, axis=0)
R_low, R_high = np.percentile(R_MC, [5, 95], axis=0)

R_nominal = simulate_collective(z1, z2)

### Plot

plt.figure(figsize=(9, 5))

plt.fill_between(
    t_adim,
    R_low / Gamma_0,
    R_high / Gamma_0,
    alpha=0.25
)

plt.plot(
    t_adim,
    R_mean / Gamma_0,
    linewidth=2,
    color='tab:blue',
    label=rf"Monte Carlo $N={N}$"
)

plt.plot(
    t_adim,
    R_nominal / Gamma_0,
    linewidth=2,
    color='tab:red',
    linestyle='--',
    label=rf"$d={d*1e9:.0f}$ nm"
)

plt.plot(
    t_adim,
    R_independent / Gamma_0,
    linewidth=1,
    color='tab:orange',
    label="Independent atoms"
)

plt.xlabel(r"$\Gamma_0 t$", fontsize=14)
plt.ylabel(r"$R/\Gamma_0$", fontsize=14)
plt.title(r"Collective emission with $\Delta z_1, \Delta z_2$ uncertainties", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show() # Figure 2