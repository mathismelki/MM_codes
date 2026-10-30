import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import time

# Global timing
t_program_start = time.perf_counter()
timings = {}

### Experimental parameters

initial_size = 16
target_size = 10

spacing = 5.0                  # Lattice spacing (µm)
filling_fraction = 0.5526      # Initial filling fraction

wavelength = 852e-9             # Wavelength (m)
waist = 1.00e-6                 # Trap waist (m)
U0_over_kB = 480e-6             # Trap depth U0/kB (K)

# Physical constants (SI units)
kB = 1.380649e-23               # Boltzmann constant (J/K)
hbar = 1.054571817e-34          # Reduced Planck constant (J.s)
m_Rb87 = 1.44316060e-25         # Mass of a 87Rb atom (kg)

# Trap depth in Joules
U0 = kB * U0_over_kB

# Optical frequency and wave number
k_opt = 2 * np.pi / wavelength
omega_opt = 2 * np.pi * 299792458 / wavelength

### Initial configuration

rng = np.random.default_rng(42)

t0 = time.perf_counter()

# Initial positions
initial_grid = (
    rng.random((initial_size, initial_size))
    < filling_fraction
)

# Coordinates centered on the optical axis
axis_initial = (
    np.arange(initial_size) - (initial_size - 1) / 2
) * spacing

X0, Y0 = np.meshgrid(axis_initial, axis_initial)

r_loaded = np.column_stack((
    X0[initial_grid],
    Y0[initial_grid]
))

# Initial trap weights: w_n^0 = 1
w_n_0 = np.ones(len(r_loaded))

# Initial optical phases: theta_n^0 ~ Uniform([0, 2*pi))
theta_n_0 = rng.uniform(
    0,
    2 * np.pi-1e-6,
    size=len(r_loaded)
)

timings["initial_configuration"] = time.perf_counter() - t0

print("Number of lattice sites:", initial_size**2)
print("Number of loaded atoms:", len(r_loaded))
print(
    "Realized filling fraction:",
    len(r_loaded) / initial_size**2
)


### Target positions: pack all loaded atoms at the grid center

axis_target = (
    np.arange(target_size) - (target_size - 1) / 2
) * spacing

X_target, Y_target = np.meshgrid(axis_target, axis_target)

r_target = np.column_stack((
    X_target.ravel(),
    Y_target.ravel()
))

### Jonker-Volgenant linear-sum assignment

# Number of initially occupied sites and target sites
N_initial = len(r_loaded)
N_target = len(r_target)

if N_initial < N_target:
    raise ValueError(
        "There are fewer loaded atoms than target sites."
    )

# Global squared-distance cost matrix (µm²)
cost_matrix = np.sum(
    (r_loaded[:, None, :] - r_target[None, :, :])**2,
    axis=2
)

t0 = time.perf_counter()

# Solve the rectangular linear assignment problem using
# SciPy's Jonker-Volgenant algorithm
atom_indices, target_indices = linear_sum_assignment(
    cost_matrix
)

# Sort assignments according to target-site index
order = np.argsort(target_indices)

atom_indices = atom_indices[order]
target_indices = target_indices[order]

timings["JV_assignment"] = time.perf_counter() - t0

# Positions of the selected atoms before rearrangement
r_initial_matched = r_loaded[atom_indices]

# Corresponding target positions
r_final_matched = r_target[target_indices]

# Convert positions to meters
r_initial_m = r_initial_matched * 1e-6
r_target_m = r_final_matched * 1e-6

# Identify atoms that are not assigned to a target site
unmatched_indices = np.setdiff1d(
    np.arange(N_initial),
    atom_indices
)

r_unmatched = r_loaded[unmatched_indices]

# Identify initially occupied sites that are selected
# and those that are not selected
occupied_selected = np.zeros(N_initial, dtype=bool)
occupied_selected[atom_indices] = True

# Global assignment cost
total_cost = np.sum(
    cost_matrix[atom_indices, target_indices]
)

# Individual transport distances
transport_distances = np.linalg.norm(
    r_final_matched - r_initial_matched,
    axis=1
)

max_distance = np.max(transport_distances)
mean_distance = np.mean(transport_distances)

print("\n--- Jonker-Volgenant assignment ---")
print("Initially occupied sites:", N_initial)
print("Target sites:", N_target)
print("Assigned atoms:", len(atom_indices))
print("Unmatched atoms:", len(unmatched_indices))
print("Global squared-distance cost:", total_cost, "µm²")
print("Mean transport distance:", mean_distance, "µm")
print("Maximum transport distance:", max_distance, "µm")


### Number of transport steps

delta_max = 0.5e-6  # Maximum displacement per step (m)

# Maximum initial-to-final transport distance
max_distance_m = np.max(
    np.linalg.norm(r_target_m - r_initial_m, axis=1)
)

# Number of transport intervals
K = int(np.ceil(max_distance_m / delta_max))

print("\n--- Transport sequence ---")
print("Maximum transport distance:", max_distance_m * 1e6, "µm")
print("Maximum displacement per step:", delta_max * 1e6, "µm")
print("Number of transport intervals K:", K)
print("Number of configurations:", K + 1)


### Physical model

def optimize_hologram(
    r_n,
    slm_size=32,
    pixel_pitch=8e-6,
    wavelength=wavelength,
    focal_length=0.01,
    magnification=1.0,
    iterations=100,
    alpha=0.35,
    seed=42
):
    """
    Optimize an SEH hologram for a given set of trap positions.

    Parameters
    ----------
    r_n : array (N, 2)
        Trap positions in the focal plane, in meters.
    slm_size : int
        Number of SLM pixels along each dimension.
    pixel_pitch : float
        SLM pixel pitch, in meters.
    wavelength : float
        Laser wavelength, in meters.
    focal_length : float
        Objective focal length, in meters.
    magnification : float
        Optical relay magnification.
    iterations : int
        Number of optimization iterations.
    alpha : float
        Weight update damping parameter.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Dictionary containing trap positions, weights, phases,
    SLM phase mask, trap fields and intensities.
    """

    r_n = np.asarray(r_n, dtype=float)
    N = len(r_n)

    if N == 0:
        raise ValueError("At least one trap is required.")
    if iterations < 1 or slm_size < 1:
        raise ValueError("iterations and slm_size must be positive.")
    if wavelength <= 0 or focal_length <= 0 or magnification <= 0:
        raise ValueError("Optical parameters must be positive.")

    rng = np.random.default_rng(seed)

    # 1. Create the SLM pixel coordinates
    axis = (
        np.arange(slm_size) - (slm_size - 1) / 2
    ) * pixel_pitch

    X, Y = np.meshgrid(axis, axis)
    r_slm = np.column_stack((X.ravel(), Y.ravel()))
    M_pixels = len(r_slm)

    # 2. Calculate the wavevector of each trap
    k_n = (
        2 * np.pi * magnification
        / (wavelength * focal_length)
    ) * r_n

    # 3. Build the propagation matrix
    # G[j, n] = exp(-i * k_n . r_slm[j])
    phase_products = r_slm @ k_n.T
    G = np.exp(-1j * phase_products)

    # 4. Initialize trap weights and random phases
    w_n = np.ones(N)
    theta_n = rng.uniform(0, 2 * np.pi, N)

    # Uniform incident beam amplitude
    A_slm = np.ones(M_pixels)

    # 5. Iterative SEH optimization
    for iteration in range(iterations):

        # Ideal complex field: U = G* c
        c_n = w_n * np.exp(1j * theta_n)
        U_ideal = G.conj() @ c_n

        # Phase-only SLM constraint
        phase_slm = np.angle(U_ideal)
        U_slm = A_slm * np.exp(1j * phase_slm)

        # Evaluate the field at the target traps
        U_F = (G.T @ U_slm) / M_pixels

        # Trap intensities
        I_n = np.abs(U_F)**2
        mean_I = np.mean(I_n)

        # Update the trap weights
        eps = 1e-12
        w_n *= (
            mean_I / np.maximum(I_n, eps)
        )**alpha

        # Normalize weights for numerical stability
        w_n /= np.mean(w_n)

        # Update target phases
        theta_n = np.angle(U_F)

    # 6. Recompute the final hologram using final parameters
    c_n = w_n * np.exp(1j * theta_n)
    U_ideal = G.conj() @ c_n

    phase_slm = np.angle(U_ideal)
    U_slm = A_slm * np.exp(1j * phase_slm)
    U_F = (G.T @ U_slm) / M_pixels
    I_n = np.abs(U_F)**2

    return {
        "r_n": r_n.copy(),
        "w_n": w_n.copy(),
        "theta_n": theta_n.copy(),
        "phase_slm": phase_slm.reshape(slm_size, slm_size),
        "U_F": U_F,
        "I_n": I_n,
        "k_n": k_n
    }


### Initial and final hologram parameters

# Initial parameters of the selected atoms
w_initial = w_n_0[atom_indices]
theta_initial = theta_n_0[atom_indices]

t0 = time.perf_counter()

# Optimize the final target hologram
result_final = optimize_hologram(
    r_target_m
)

# Final weights and phases
w_final = result_final["w_n"]
theta_final = result_final["theta_n"]

timings["final_hologram_optimization"] = time.perf_counter() - t0


### Generate intermediate transport holograms

def generate_transport_frames(
    r_initial,
    r_final,
    w_initial,
    w_final,
    theta_initial,
    theta_final,
    K,
    slm_size=32,
    pixel_pitch=8e-6,
    wavelength=852e-9,
    focal_length=0.01,
    magnification=1.0,
    optimize_weights=True,
    alpha=0.35
):
    """
    Generate intermediate holograms using Eqs. (14)-(20).

    Positions are expressed in meters.
    """

    N = len(r_initial)

    # Check input dimensions
    if r_initial.shape != r_final.shape:
        raise ValueError("Initial and final positions must have the same shape.")

    if len(w_initial) != N or len(w_final) != N:
        raise ValueError("Weight arrays must have length N.")

    if len(theta_initial) != N or len(theta_final) != N:
        raise ValueError("Phase arrays must have length N.")

    if K < 1:
        raise ValueError("K must be at least 1.")

    # SLM coordinates
    axis = (
        np.arange(slm_size) - (slm_size - 1) / 2
    ) * pixel_pitch

    X, Y = np.meshgrid(axis, axis)

    r_slm = np.column_stack((X.ravel(), Y.ravel()))

    M_pixels = len(r_slm)

    # Optical scaling factor
    optical_factor = (
        2 * np.pi * magnification
        / (wavelength * focal_length)
    )

    # Shortest phase difference: Eq. (18)
    delta_theta = (
        (theta_final - theta_initial + np.pi)
        % (2 * np.pi)
    ) - np.pi

    # Storage for all transport frames
    frames = []

    # Frame times
    frame_times = []

    # Generate frames p = 0, ..., K
    for p in range(K + 1):

        t_frame_start = time.perf_counter()

        s = p / K

        # Linear position interpolation
        r_p = (
            (1 - s) * r_initial
            + s * r_final
        )

        # Trap wavevectors
        k_p = optical_factor * r_p

        # Linear weight interpolation
        w_p = (
            (1 - s) * w_initial
            + s * w_final
        )

        # Phase interpolation
        theta_p = theta_initial + s * delta_theta

        # Ideal complex field at the SLM
        phase_products = r_slm @ k_p.T

        U_ideal = np.exp(
            1j * (phase_products + theta_p[None, :])
        ) @ w_p

        # Phase-only SLM constraint
        phase_slm = np.angle(U_ideal)

        U_slm = np.exp(1j * phase_slm)

        # Calculate the field at the trap positions
        G_p = np.exp(-1j * phase_products)

        U_F = (G_p.T @ U_slm) / M_pixels

        # Trap intensities
        I_p = np.abs(U_F)**2

        # Optional single weight-update iteration
        if optimize_weights and 0 < p < K:

            mean_I = np.mean(I_p)
            eps = 1e-12

            w_p = w_p * (
                mean_I / np.maximum(I_p, eps)
            )**alpha

            # Normalize weights
            w_p /= np.mean(w_p)

            # Recompute the hologram with updated weights
            U_ideal = np.exp(
                1j * (phase_products + theta_p[None, :])
            ) @ w_p

            phase_slm = np.angle(U_ideal)

            U_slm = np.exp(1j * phase_slm)

            U_F = (G_p.T @ U_slm) / M_pixels

            I_p = np.abs(U_F)**2

        frame_times.append(time.perf_counter() - t_frame_start)

        # Save the frame
        frames.append({
            "p": p,
            "s": s,
            "r_n": r_p,
            "k_n": k_p,
            "w_n": w_p,
            "theta_n": theta_p,
            "phase_slm": phase_slm.reshape(
                slm_size, slm_size
            ),
            "U_F": U_F,
            "I_n": I_p
        })

    print("\n--- Timing of intermediate holograms ---")
    print("Mean time per frame:", np.mean(frame_times) * 1000, "ms")
    print("Maximum time per frame:", np.max(frame_times) * 1000, "ms")
    print("Total frame-generation time:", np.sum(frame_times) * 1000, "ms")

    return frames

### Generate the transport sequence

t0 = time.perf_counter()

frames = generate_transport_frames(
    r_initial=r_initial_m,
    r_final=r_target_m,
    w_initial=w_initial,
    w_final=w_final,
    theta_initial=theta_initial,
    theta_final=theta_final,
    K=K,
    wavelength=wavelength,
    optimize_weights=True,
    alpha=0.35
)

timings["intermediate_holograms"] = time.perf_counter() - t0

print("\nNumber of generated frames:", len(frames))

# Inspect the first and last frames
print("Initial frame:", frames[0]["p"])
print("Final frame:", frames[-1]["p"])


### Plots

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ==========================================================
# FIGURE 1 : JV assignment and transport trajectories
# ==========================================================

ax1.scatter(
    r_loaded[:, 0],
    r_loaded[:, 1],
    color="lightgray",
    s=35,
    label="Initially occupied sites",
    zorder=2
)

ax1.scatter(
    r_initial_matched[:, 0],
    r_initial_matched[:, 1],
    color="royalblue",
    s=40,
    edgecolors="black",
    label="Selected atoms",
    zorder=3
)

ax1.scatter(
    r_unmatched[:, 0],
    r_unmatched[:, 1],
    color="gray",
    marker="x",
    s=50,
    label="Unmatched atoms",
    zorder=3
)

ax1.scatter(
    r_target[:, 0],
    r_target[:, 1],
    color="tomato",
    marker="s",
    s=35,
    edgecolors="black",
    label="Target sites",
    zorder=4
)

for i in range(N_target):
    ax1.annotate(
        "",
        xy=r_final_matched[i],
        xytext=r_initial_matched[i],
        arrowprops=dict(
            arrowstyle="->",
            color="royalblue",
            alpha=0.45,
            linewidth=1
        )
    )

ax1.set_aspect("equal")

margin = spacing / 2

ax1.set_xlim(
    axis_initial[0] - margin,
    axis_initial[-1] + margin
)

ax1.set_ylim(
    axis_initial[-1] + margin,
    axis_initial[0] - margin
)

ax1.set_xticks(axis_initial)
ax1.set_yticks(axis_initial)

ax1.grid(True, alpha=0.35, zorder=0)
ax1.tick_params(
    axis="both",
    which="both",
    labelbottom=False,
    labelleft=False
)

ax1.set_axisbelow(True)
ax1.set_xlabel("x (µm)")
ax1.set_ylabel("y (µm)")
ax1.set_title(rf"(a) Atom assignment and transport trajectories ({spacing} $\mu$m increments)")
ax1.legend(
    loc="best",
    fontsize=9,
    frameon=True,
    framealpha=1.0,
    labelcolor="black"
)


# ==========================================================
# FIGURE 2 : Intermediate atom configurations
# ==========================================================

ax2.scatter(
    r_loaded[:, 0],
    r_loaded[:, 1],
    color="lightgray",
    s=25,
    label="Initial occupied sites",
    zorder=2
)

ax2.scatter(
    r_target[:, 0],
    r_target[:, 1],
    color="tomato",
    marker="s",
    s=35,
    label="Target sites",
    zorder=2
)

for p in range(1, K):
    positions_um = frames[p]["r_n"] * 1e6

    ax2.scatter(
        positions_um[:, 0],
        positions_um[:, 1],
        color="royalblue",
        s=12,
        alpha=0.25
    )

ax2.set_xlim(
    axis_initial[0] - margin,
    axis_initial[-1] + margin
)

ax2.set_ylim(
    axis_initial[-1] + margin,
    axis_initial[0] - margin
)

ax2.set_xticks(axis_initial)
ax2.set_yticks(axis_initial)

ax2.grid(True, alpha=0.35)

ax2.tick_params(
    axis="both",
    which="both",
    labelbottom=False,
    labelleft=False
)

ax2.set_aspect("equal")
ax2.set_xlabel("x (µm)")
ax2.set_ylabel("y (µm)")
ax2.set_title(rf"(b) Intermediate atom configurations ({spacing} $\mu$m increments)")

ax2.legend(
    loc="upper left",
    fontsize=9,
    frameon=True,
    framealpha=1.0,
    labelcolor="black"
)

#----------------------------------------------------------

print("\n===================================")
print("       TEMPORAL PERFORMANCE")
print("===================================")

for name, duration in timings.items():
    print(f"{name:35s}: {duration * 1000:.3f} ms")

total_computation_time = sum(timings.values())

print("-----------------------------------")
print(
    f"{'Total measured computation':35s}: "
    f"{total_computation_time * 1000:.3f} ms"
)

print(f"{'Number of generated frames':35s}: {len(frames)}")

print("-----------------------------------")
print("Reference article:")
print("Mean number of intermediate holograms: 20.78")
print("Mean rearrangement time: 18.23 ms")
print("===================================")

#----------------------------------------------------------

plt.tight_layout()
plt.show()