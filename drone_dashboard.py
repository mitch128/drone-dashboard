import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
import math
from scipy.interpolate import CubicSpline
import folium
from streamlit_folium import folium_static
import folium

#######################################
# 1. Generate More Realistic Dummy Drone Data in 3D
#######################################
def make_dummy_data():
    np.random.seed(42)
    timestamps = np.arange(0, 61)  # Extended to 60 seconds for better visualization
    rows = []

    # Drone D1: High-threat (Shahed) with a curved 3D trajectory. It descends as it approaches.
    t_D1 = np.linspace(0, 60, 100)
    x_D1 = CubicSpline([0, 15, 30, 45, 60], [-600, -400, -200, 0, 200])(t_D1)
    y_D1 = CubicSpline([0, 15, 30, 45, 60], [-500, -300, -100, 100, 300])(t_D1)
    z_D1 = CubicSpline([0, 15, 30, 45, 60], [200, 150, 100, 50, 0])(t_D1)
    for t in timestamps:
        idx = np.searchsorted(t_D1, t)
        pos = np.array([x_D1[idx], y_D1[idx], z_D1[idx]])
        vel = np.array([np.nan, np.nan, np.nan]) if t == 0 else np.array([x_D1[idx] - x_D1[idx-1], y_D1[idx] - y_D1[idx-1], z_D1[idx] - z_D1[idx-1]]) / (t_D1[idx] - t_D1[idx-1])
        rows.append({
            'time': t,
            'id': 'D1',
            'type': 'Shahed',
            'x': pos[0],
            'y': pos[1],
            'z': pos[2],
            'velocity_x': vel[0],
            'velocity_y': vel[1],
            'velocity_z': vel[2]
        })

    # Drone D2: Recon drone (DJI Mavic) following a looping spiral in 3D
    t_D2 = np.linspace(0, 60, 100)
    angle_D2 = 0.2 * t_D2
    radius_D2 = 50 + 0.5 * t_D2
    x_D2 = 200 + radius_D2 * np.cos(angle_D2)
    y_D2 = 400 + radius_D2 * np.sin(angle_D2)
    z_D2 = 100 + 2 * t_D2
    for t in timestamps:
        idx = np.searchsorted(t_D2, t)
        pos = np.array([x_D2[idx], y_D2[idx], z_D2[idx]])
        vel = np.array([np.nan, np.nan, np.nan]) if t == 0 else np.array([x_D2[idx] - x_D2[idx-1], y_D2[idx] - y_D2[idx-1], z_D2[idx] - z_D2[idx-1]]) / (t_D2[idx] - t_D2[idx-1])
        rows.append({
            'time': t,
            'id': 'D2',
            'type': 'DJI Mavic',
            'x': pos[0],
            'y': pos[1],
            'z': pos[2],
            'velocity_x': vel[0],
            'velocity_y': vel[1],
            'velocity_z': vel[2]
        })

    # Drone D3: Surveillance drone (Recon) with a steady drift while oscillating in z
    t_D3 = np.linspace(0, 60, 100)
    x_D3 = -300 + 10 * t_D3
    y_D3 = 600 - 12 * t_D3
    z_D3 = 150 + 1 * t_D3 + 3 * np.sin(0.2 * t_D3)
    for t in timestamps:
        idx = np.searchsorted(t_D3, t)
        pos = np.array([x_D3[idx], y_D3[idx], z_D3[idx]])
        vel = np.array([np.nan, np.nan, np.nan]) if t == 0 else np.array([x_D3[idx] - x_D3[idx-1], y_D3[idx] - y_D3[idx-1], z_D3[idx] - z_D3[idx-1]]) / (t_D3[idx] - t_D3[idx-1])
        rows.append({
            'time': t,
            'id': 'D3',
            'type': 'Recon',
            'x': pos[0],
            'y': pos[1],
            'z': pos[2],
            'velocity_x': vel[0],
            'velocity_y': vel[1],
            'velocity_z': vel[2]
        })

    df = pd.DataFrame(rows)
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
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

#######################################
# Utility: Compute Projected Impact Zone for High-threat Drone in 3D
#######################################
def compute_impact_zone(x, y, z, vx, vy, vz, seconds=5):
    proj_x = x + vx * seconds
    proj_y = y + vy * seconds
    proj_z = z + vz * seconds
    displacement = math.sqrt((vx*seconds)**2 + (vy*seconds)**2 + (vz*seconds)**2)
    impact_radius = 0.1 * displacement + 20
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

    xx, yy = np.meshgrid(np.linspace(-700,700,2), np.linspace(-700,700,2))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color='lightgray', alpha=0.3)

    for name, (ix, iy, iz) in infantry_positions.items():
        ax.scatter(ix, iy, iz, c='k', marker='^', s=100)
        ax.text(ix+10, iy+10, iz+10, name, fontsize=10, color='black')

    frame = df[df.time == t]
    drone_info = {}
    events = []
    for _, row in frame.iterrows():
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
        ax.text(row.x+10, row.y+10, row.z+10, f"{row['id']}\n{drone_type}\nV:{vel_mag:.1f}", fontsize=9, color=color)

        if drone_type == 'Shahed':
            vx, vy, vz = row.velocity_x, row.velocity_y, row.velocity_z
            proj_x, proj_y, proj_z, impact_radius = compute_impact_zone(row.x, row.y, row.z, vx, vy, vz, seconds=5)
            ax.quiver(row.x, row.y, row.z, proj_x-row.x, proj_y-row.y, proj_z-row.z,
                      color='red', length=1, normalize=False, arrow_length_ratio=0.15, alpha=0.6)
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            xs = proj_x + impact_radius * np.outer(np.cos(u), np.sin(v))
            ys = proj_y + impact_radius * np.outer(np.sin(u), np.sin(v))
            zs = proj_z + impact_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_wireframe(xs, ys, zs, color='red', alpha=0.3)
            ax.text(proj_x+10, proj_y+10, proj_z+10, "Impact Zone", fontsize=8, color='red')
            events.append(f"ALERT: {row['id']} [{drone_type}] projected impact zone at ({proj_x:.1f}, {proj_y:.1f}, {proj_z:.1f}) with radius {impact_radius:.1f}m")

    return fig, frame, drone_info, events

#######################################
# 4. Streamlit UI – Command Center Dashboard for 3D Visualization
#######################################
st.set_page_config(page_title="Drone Intelligence Dashboard (3D)", layout="wide")
st.title("Drone Intelligence Dashboard (3D)")
st.markdown("""
This tool monitors enemy drone activity in a 3D space relative to frontline units.
Key data includes position (x, y, z), time, and velocity.
Use the control panel to play or step through the simulation.
""")

# Sidebar Controls
st.sidebar.header("Command Center Controls")
t_slider = st.sidebar.slider("Select Time (s)", 0, int(df.time.max()), 0, 1, key="time_slider")
play = st.sidebar.button("Play Live Simulation")
update_interval = st.sidebar.number_input("Simulation Speed (seconds per frame)", min_value=0.1, max_value=5.0, value=0.75, step=0.1)

# Layout: left for 3D radar display, right for summary and event log
col_radar, col_summary = st.columns([2, 1])
with col_radar:
    radar_placeholder = st.empty()
with col_summary:
    st.subheader("Battlefield Summary")
    summary_placeholder = st.empty()
    st.subheader("Event Log")
    event_placeholder = st.empty()

def generate_summary(frame, drone_info):
    summary_lines = []
    summary_lines.append("**Current Drone Counts:**")
    for drone_type, count in drone_info.items():
        summary_lines.append(f"- {drone_type}: {count}")
    summary_lines.append("\n**Nearest Drone per Unit:**")
    for name, pos in infantry_positions.items():
        min_dist = float("inf")
        closest_drone = None
        for _, row in frame.iterrows():
            d = distance_3d(pos, (row.x, row.y, row.z))
            if d < min_dist:
                min_dist = d
                closest_drone = f"{row['type']} ({row['id']})"
        if closest_drone:
            summary_lines.append(f"- {name}: {closest_drone} at {min_dist:.1f} m")
        else:
            summary_lines.append(f"- {name}: No drone detected")
    return "\n".join(summary_lines)

#######################################
# 5. Real-time Simulation / Slider Update Handling
#######################################
if play:
    max_time = int(df.time.max())
    for t in range(t_slider, max_time + 1):
        fig, frame, drone_info, events = plot_radar_frame_3d(t)
        radar_placeholder.pyplot(fig)
        summary_placeholder.markdown(generate_summary(frame, drone_info))
        if events:
            event_placeholder.markdown("\n".join([f"- {e}" for e in events]))
        else:
            event_placeholder.markdown("No new alerts at this time.")
        time.sleep(update_interval)
    st.session_state["time_slider"] = max_time
else:
    fig, frame, drone_info, events = plot_radar_frame_3d(t_slider)
    radar_placeholder.pyplot(fig)
    summary_placeholder.markdown(generate_summary(frame, drone_info))
    if events:
        event_placeholder.markdown("\n".join([f"- {e}" for e in events]))
    else:
        event_placeholder.markdown("No critical events at this time.")

#######################################
# 6. Map Overlay
#######################################
st.sidebar.header("Map Overlay")
show_map = st.sidebar.checkbox("Show Map Overlay", value=True)

if show_map:
    st.subheader("Map Overlay")
    m = folium.Map(location=[51.7309, 36.1858], zoom_start=12)  # Kursk, Russia coordinates
    for name, (x, y, z) in infantry_positions.items():
        folium.Marker([51.7309 + y/10000, 36.1858 + x/10000], popup=name).add_to(m)
    folium_static(m)
