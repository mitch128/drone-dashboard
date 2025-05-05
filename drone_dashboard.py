import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
import math

#######################################
# 1. Generate More Realistic Dummy Drone Data in 3D
#######################################
def make_dummy_data():
    np.random.seed(42)
    timestamps = np.arange(0, 31)  # Simulate over 30 seconds
    rows = []

    # Drone D1: High-threat (Shahed) with a curved, descending 3D trajectory.
    start_D1 = np.array([-600, -500, 200])
    base_velocity_D1 = np.array([30, 35, -4])
    for t in timestamps:
        deviation = np.array([15 * np.sin(0.1 * t), 15 * np.cos(0.1 * t), 5 * np.sin(0.05 * t)])
        noise = np.random.normal(0, 2, 3)
        pos = start_D1 + base_velocity_D1 * t + deviation + noise
        rows.append({
            'time': t,
            'id': 'D1',
            'type': 'Shahed',
            'x': pos[0],
            'y': pos[1],
            'z': pos[2]
        })

    # Drone D2: Recon drone (DJI Mavic) following a looping 3D spiral with upward drift.
    start_D2 = np.array([200, 400, 100])
    for t in timestamps:
        angle = 0.2 * t
        radius = 50 + 0.5 * t  # outward spiral
        pos = start_D2 + np.array([radius * np.cos(angle), radius * np.sin(angle), 2 * t])
        noise = np.random.normal(0, 1.5, 3)
        pos = pos + noise
        rows.append({
            'time': t,
            'id': 'D2',
            'type': 'DJI Mavic',
            'x': pos[0],
            'y': pos[1],
            'z': pos[2]
        })

    # Drone D3: Surveillance drone (Recon) with steady drift and oscillatory vertical movement.
    start_D3 = np.array([-300, 600, 150])
    for t in timestamps:
        base = start_D3 + np.array([10 * t, -12 * t, 1 * t])
        oscillation = np.array([8 * np.sin(0.15 * t), 8 * np.cos(0.15 * t), 3 * np.sin(0.2 * t)])
        noise = np.random.normal(0, 1, 3)
        pos = base + oscillation + noise
        rows.append({
            'time': t,
            'id': 'D3',
            'type': 'Recon',
            'x': pos[0],
            'y': pos[1],
            'z': pos[2]
        })

    df = pd.DataFrame(rows)

    # Compute approximate velocity using backward differences for each drone.
    for drone_id in df['id'].unique():
        drone_mask = df['id'] == drone_id
        df.loc[drone_mask, ['velocity_x', 'velocity_y', 'velocity_z']] = df[drone_mask][['x', 'y', 'z']].diff().fillna(0)

    return df

df = make_dummy_data()

#######################################
# 2. Define Friendly Positions (Frontline Units) in 3D
#######################################
infantry_positions = {
    "Alpha (1st Battalion HQ)": (0, 0, 0),
    "Bravo (FOB – Forward Operating Base)": (250, 150, 0),
    "Charlie (Observation Post)": (-200, -150, 0)
}

#######################################
# Utility: Euclidean Distance Calculation in 3D
#######################################
def distance_3d(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

#######################################
# Utility: Compute Projected Impact Zone for High-threat Drone in 3D
#######################################
def compute_impact_zone(x, y, z, vx, vy, vz, seconds=5):
    proj_x = x + vx * seconds
    proj_y = y + vy * seconds
    proj_z = z + vz * seconds
    displacement = math.sqrt((vx * seconds)**2 + (vy * seconds)**2 + (vz * seconds)**2)
    impact_radius = 0.15 * displacement + 15
    return (proj_x, proj_y, proj_z, impact_radius)

#######################################
# 3. 3D Plotting Function for a Single Timeframe
#######################################
def plot_radar_frame_3d(t):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#F7F7F7')
    ax.set_title(f"3D Drone Tracker – Time = {t}s", fontsize=16)

    ax.set_xlim(-700, 700)
    ax.set_ylim(-700, 700)
    ax.set_zlim(0, 400)

    # Draw the ground plane (simulate terrain)
    xx, yy = np.meshgrid(np.linspace(-700, 700, 2), np.linspace(-700, 700, 2))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color='lightgray', alpha=0.3)

    # Plot friendly unit positions (ground units)
    for name, (ix, iy, iz) in infantry_positions.items():
        ax.scatter(ix, iy, iz, c='k', marker='^', s=100)
        ax.text(ix + 10, iy + 10, iz + 10, name, fontsize=10, color='black', weight='bold')

    # Get current frame data
    current_frame = df[df.time == t]
    drone_info = {}
    events = []

    # For each drone, plot its complete trajectory up till time t.
    for drone_id in df['id'].unique():
        history = df[(df.id == drone_id) & (df.time <= t)]
        ax.plot(history.x, history.y, history.z, linestyle='dotted', color='gray', alpha=0.7)

    # Plot the current positions and annotations
    for _, row in current_frame.iterrows():
        drone_type = row['type']
        drone_info[drone_type] = drone_info.get(drone_type, 0) + 1

        if drone_type == 'Shahed':
            color = 'red'
        elif drone_type == 'DJI Mavic':
            color = 'blue'
        else:
            color = 'green'

        ax.scatter(row.x, row.y, row.z, c=color, marker='o', s=80, alpha=0.9)

        vel_mag = math.sqrt(row.velocity_x**2 + row.velocity_y**2 + row.velocity_z**2)
        annotation = f"{row['id']}\n{drone_type}\nV:{vel_mag:.1f} m/s"
        ax.text(row.x + 10, row.y + 10, row.z + 10, annotation, fontsize=9, color=color)

        # For the high-threat 'Shahed' drone, compute and plot the projected impact zone.
        if drone_type == 'Shahed':
            vx, vy, vz = row.velocity_x, row.velocity_y, row.velocity_z
            proj_x, proj_y, proj_z, impact_radius = compute_impact_zone(row.x, row.y, row.z, vx, vy, vz, seconds=5)
            ax.quiver(row.x, row.y, row.z, proj_x - row.x, proj_y - row.y, proj_z - row.z,
                      color='red', arrow_length_ratio=0.15, alpha=0.7)
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            xs = proj_x + impact_radius * np.outer(np.cos(u), np.sin(v))
            ys = proj_y + impact_radius * np.outer(np.sin(u), np.sin(v))
            zs = proj_z + impact_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_wireframe(xs, ys, zs, color='red', alpha=0.3)
            ax.text(proj_x + 10, proj_y + 10, proj_z + 10, "Projected\nImpact Zone", fontsize=8, color='red')
            events.append(f"ALERT: {row['id']} (Shahed) estimated impact at ({proj_x:.1f}, {proj_y:.1f}, {proj_z:.1f}) with radius {impact_radius:.1f} m")

    return fig, current_frame, drone_info, events

#######################################
# 4. 2D Birds-eye Plotting Function for a Single Timeframe
#######################################
def plot_radar_frame_2d(t):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_facecolor('#F7F7F7')
    ax.set_title(f"2D Radar Tracker – Time = {t}s", fontsize=16)

    ax.set_xlim(-700, 700)
    ax.set_ylim(-700, 700)

    # Plot friendly unit positions on ground plane
    for name, (ix, iy, iz) in infantry_positions.items():
        ax.scatter(ix, iy, c='k', marker='^', s=100)
        ax.text(ix + 10, iy + 10, name, fontsize=10, color='black', weight='bold')

    # Plot drones
    current_frame = df[df.time == t]
    for _, row in current_frame.iterrows():
        color = 'red' if row['type'] == 'Shahed' else 'blue' if row['type'] == 'DJI Mavic' else 'green'
        ax.scatter(row.x, row.y, c=color, marker='o', s=80, alpha=0.9)
        ax.text(row.x + 10, row.y + 10, f"{row['id']} ({row['type']})", fontsize=10, color=color)

    return fig

#######################################
# 5. Streamlit Interface for Real-Time Simulation
#######################################
st.title("Drone Tracker Simulation")
time_slider = st.slider("Select Time (seconds)", 0, 30, 0)
mode = st.radio("Choose View Mode", ['3D', '2D'])

# Generate either 2D or 3D plot based on user selection
if mode == '3D':
    fig, current_frame, drone_info, events = plot_radar_frame_3d(time_slider)
    st.pyplot(fig)
elif mode == '2D':
    fig = plot_radar_frame_2d(time_slider)
    st.pyplot(fig)

# Show event logs and drone info
st.write("### Event Log:")
if 'ALERT' in events:
    st.markdown("\n".join(events))

st.write(f"### Drone Information: {drone_info}")
