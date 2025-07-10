import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Define the PBRS Hyperparameters ---
# Using the "average" baseline parameters for this visualization
D_SAFE = 25.0
K_BUBBLE = 105.0
K_DECAY = 0.1
K_POS = 0.01
W_SAFE = 0.5
W_POS = 0.5
R_COLL = 6.5 # agent_size/2 + intruder_size/2 = 5 + 1.5

# --- 2. Define the Scenario to Visualize ---
ENV_SIZE = 100
TARGET_POS = np.array([90, 90])
# Let's place a few intruders in interesting positions
INTRUDER_POSITIONS = [
    np.array([30, 70]),
    np.array([60, 50]),
    np.array([80, 20])
]

# --- 3. Define the Potential Functions (from your env code) ---

def potential_pos(agent_pos):
    """Calculates the potential based on distance to the target."""
    dist_sq = np.sum((agent_pos - TARGET_POS)**2)
    return -K_POS * dist_sq

def potential_safe(agent_pos):
    """Calculates the potential based on the safety bubble concept."""
    total_safe_potential = 0.0
    for intruder_pos in INTRUDER_POSITIONS:
        dist = np.linalg.norm(agent_pos - intruder_pos)
        if dist < D_SAFE:
            total_safe_potential += -K_BUBBLE * np.exp(-K_DECAY * (dist - R_COLL))
    return total_safe_potential

def total_potential(agent_pos):
    """Calculates the total potential of a state."""
    phi_p = potential_pos(agent_pos)
    phi_s = potential_safe(agent_pos)
    return W_POS * phi_p + W_SAFE * phi_s

# --- 4. Generate Data for the Plots ---

# Create a grid of points to evaluate the potential function on
grid_resolution = 200
x = np.linspace(0, ENV_SIZE, grid_resolution)
y = np.linspace(0, ENV_SIZE, grid_resolution)
xx, yy = np.meshgrid(x, y)
positions = np.c_[xx.ravel(), yy.ravel()]

# Calculate the potential at each point on the grid
z = np.array([total_potential(p) for p in positions])
zz = z.reshape(xx.shape)

# --- 5. Create the Visualization ---

# Use a professional plot style
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Create the contour plot (heatmap)
contour = ax.contourf(xx, yy, zz, levels=50, cmap='viridis_r') # _r reverses the colormap

# Add a color bar to show the potential values
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label('Potential Φ(s) (Lower is Better)', rotation=270, labelpad=20, fontsize=12)

# Plot the target position (as a star)
ax.plot(TARGET_POS[0], TARGET_POS[1], '*', color='gold', markersize=20, markeredgecolor='black', label='Target')

# Plot the intruder positions and their safety bubbles
for i, pos in enumerate(INTRUDER_POSITIONS):
    ax.plot(pos[0], pos[1], 'o', color='red', markersize=12, label='Intruder' if i == 0 else "")
    # Draw the d_safe circle
    safety_circle = plt.Circle((pos[0], pos[1]), D_SAFE, color='red', fill=False, linestyle='--', alpha=0.5)
    ax.add_artist(safety_circle)

# --- Formatting ---
ax.set_xlabel('X-Position (meters)', fontsize=12)
ax.set_ylabel('Y-Position (meters)', fontsize=12)
ax.set_title('PBRS Potential Field Landscape', fontsize=16, pad=20)
ax.legend(fontsize=12)
ax.set_aspect('equal', adjustable='box')
plt.xlim(0, ENV_SIZE)
plt.ylim(0, ENV_SIZE)

# Save the figure
output_dir = "plots/"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "pbrs_landscape_visualization.png"), bbox_inches='tight', dpi=300)

print(f"✅ Visualization saved to {os.path.join(output_dir, 'pbrs_landscape_visualization.png')}")

plt.show()