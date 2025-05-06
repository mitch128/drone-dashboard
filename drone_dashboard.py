import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D projection
import time
import math

#######################################
# 1. Generate More Realistic Dummy Drone Data in 3D
#######################################
def make_dummy_data():
    # We'll simulate 3 drones (each with id and type) for a 30-sec simulation in 3D (x, y, z)
    np.random.seed(42)  # reproducible data
    timestamps = np.arange(0, 31)  # 0 to 30 seconds
    rows = []
    
    # Drone D1: High-threat (Shahed) with a curved, descending 3D trajectory.
    start_D1 = np.array([-600, -500, 200])
    base_velocity_D1 = np.array([30, 35, -4])
    for t in timestamps:
        deviation = np.array([15 * np.sin(0.1*t), 15 * np.cos(0.1*t), 5 * np.sin(0.05*t)])
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
        pos = start_D2 + np.array([radius * np.cos(angle), radius * np.sin(angle), 2*t])
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
        base = start_D3 + np.array([10*t, -12*t, 1*t])
        oscillation = np.array([8 * np.sin(0.15*t), 8 * np.cos(0.15*t), 3*np.sin(0.2*t)])
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
# Assume friendly units are on the ground (z=0)
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
    # Extrapolate future position in 3D based on current velocity
    proj_x = x + vx * seconds
    proj_y = y + vy * seconds
    proj_z = z + vz * seconds
    # To simulate uncertainty, add a factor proportional to velocity magnitude and seconds.
    displacement = math.sqrt((vx * seconds)**2 + (vy * seconds)**2 + (vz * seconds)**2)
    # Uncertainty increases slightly over time; use 15% factor plus a minimum uncertainty.
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
    
    # Set the plot limits
    ax.set_xlim(-700, 700)
    ax.set_ylim(-700, 700)
    ax.set_zlim(0, 400)
    
    # Draw the ground plane (simulate terrain) as a semi-transparent surface.
    xx, yy = np.meshgrid(np.linspace(-700,700,2), np.linspace(-700,700,2))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color='lightgray', alpha=0.3)
    
    # Plot friendly unit positions (ground units)
    for name, (ix, iy, iz) in infantry_positions.items():
        ax.scatter(ix, iy, iz, c='k', marker='^', s=100)
        ax.text(ix+10, iy+10, iz+10, name, fontsize=10, color='black', weight='bold')
    
    # Get current frame data and also historical data for dotted trajectories.
    current_frame = df[df.time == t]
    drone_info = {}
    events = []
    
    # For each drone, plot its complete trajectory up till time t.
    for drone_id in df['id'].unique():
        history = df[(df.id == drone_id) & (df.time <= t)]
        ax.plot(history.x, history.y, history.z, linestyle='dotted', color='gray', alpha=0.7)
    
    # Now, plot the current positions and annotations
    for _, row in current_frame.iterrows():
        drone_type = row['type']
        drone_info[drone_type] = drone_info.get(drone_type, 0) + 1
        
        if drone_type == 'Shahed':
            color = 'red'
        elif drone_type == 'DJI Mavic':
            color = 'blue'
        else:
            color = 'green'
        
        # Plot the current position
        ax.scatter(row.x, row.y, row.z, c=color, marker='o', s=80, alpha=0.9)
        
        # Annotate with ID, type, velocity magnitude
        vel_mag = math.sqrt(row.velocity_x**2 + row.velocity_y**2 + row.velocity_z**2)
        annotation = f"{row['id']}\n{drone_type}\nV:{vel_mag:.1f} m/s"
        ax.text(row.x+10, row.y+10, row.z+10, annotation, fontsize=9, color=color)
        
        # For the high-threat 'Shahed' drone, compute and plot the projected impact zone.
        if drone_type == 'Shahed':
            vx, vy, vz = row.velocity_x, row.velocity_y, row.velocity_z
            proj_x, proj_y, proj_z, impact_radius = compute_impact_zone(row.x, row.y, row.z, vx, vy, vz, seconds=5)
            # Draw an arrow indicating projected movement.
            ax.quiver(row.x, row.y, row.z, proj_x-row.x, proj_y-row.y, proj_z-row.z,
                      color='red', arrow_length_ratio=0.15, alpha=0.7)
            # Draw projected impact sphere (as a wireframe)
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            xs = proj_x + impact_radius * np.outer(np.cos(u), np.sin(v))
            ys = proj_y + impact_radius * np.outer(np.sin(u), np.sin(v))
            zs = proj_z + impact_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_wireframe(xs, ys, zs, color='red', alpha=0.3)
            ax.text(proj_x+10, proj_y+10, proj_z+10, "Projected\nImpact Zone", fontsize=8, color='red')
            events.append(f"ALERT: {row['id']} (Shahed) estimated impact at ({proj_x:.1f}, {proj_y:.1f}, {proj_z:.1f}) with radius {impact_radius:.1f} m")
    
    return fig, current_frame, drone_info, events

#######################################
# 4. 2D Birds-eye Plotting Function for a Single Timeframe
#######################################
def plot_radar_frame_2d(t):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_facecolor('#F7F7F7')
    ax.set_title(f"2D Birds-eye Drone Tracker – Time = {t}s", fontsize=16)
    
    # Set limits (assuming a flat map view)
    ax.set_xlim(-700, 700)
    ax.set_ylim(-700, 700)
    
    # Draw concentric engagement circles (100m, 250m, 500m) around each friendly unit.
    for name, pos in infantry_positions.items():
        ix, iy, _ = pos
        for r in (100, 250, 500):
            circle = plt.Circle((ix, iy), r, color='gray', fill=False, linestyle='dotted', alpha=0.5)
            ax.add_patch(circle)
    
    # Plot friendly unit positions
    for name, (ix, iy, _) in infantry_positions.items():
        ax.plot(ix, iy, 'ks', markersize=10)
        ax.text(ix+10, iy+10, name, fontsize=10, color='black', weight='bold')
    
    # Get current frame data and historical trajectories for each drone.
    current_frame = df[df.time == t]
    
    for drone_id in df['id'].unique():
        history = df[(df.id == drone_id) & (df.time <= t)]
        ax.plot(history.x, history.y, linestyle='dotted', color='gray', alpha=0.7)
    
    # Plot current drone positions and annotate with details
    drone_info = {}
    events = []
    for _, row in current_frame.iterrows():
        drone_type = row['type']
        drone_info[drone_type] = drone_info.get(drone_type, 0) + 1
        
        if drone_type == 'Shahed':
            color = 'red'
        elif drone_type == 'DJI Mavic':
            color = 'blue'
        else:
            color = 'green'
        
        ax.plot(row.x, row.y, 'o', color=color, markersize=8, alpha=0.9)
        vel_mag = math.sqrt(row.velocity_x**2 + row.velocity_y**2 + row.velocity_z**2)
        ax.text(row.x+10, row.y+10, f"{row['id']}\nV:{vel_mag:.1f}", fontsize=8, color=color)
        
        # For high-threat drones, indicate projected impact zone simplified in 2D.
        if drone_type == 'Shahed':
            vx, vy = row.velocity_x, row.velocity_y
            proj_x = row.x + vx * 5
            proj_y = row.y + vy * 5
            displacement = math.sqrt((vx * 5)**2 + (vy * 5)**2)
            impact_radius = 0.15 * displacement + 15
            ax.arrow(row.x, row.y, proj_x-row.x, proj_y-row.y, head_width=15, head_length=15,
                     fc='red', ec='red', alpha=0.7)
            impact_circle = plt.Circle((proj_x, proj_y), impact_radius, fill=True, color='red', alpha=0.2)
            ax.add_patch(impact_circle)
            ax.text(proj_x+10, proj_y+10, "Impact Zone", fontsize=8, color='red')
            events.append(f"ALERT: {row['id']} (Shahed) 2D impact at ({proj_x:.1f}, {proj_y:.1f}) with radius {impact_radius:.1f} m")
    
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig, current_frame, drone_info, events

#######################################
# 5. Streamlit UI – Combined Dashboard (3D and 2D)
#######################################
st.set_page_config(page_title="Drone Intelligence Dashboard (Multi-View)", layout="wide")
st.title("Drone Intelligence Dashboard (Multi-View)")
st.markdown("""
This tool monitors enemy drone activity in both 3D space and on a 2D birds‑eye view relative to frontline units.  
Key data includes position (x, y, z), time, and velocity.  
Use the control panel to play or step through the simulation.
""")

# Sidebar Controls
st.sidebar.header("Command Center Controls")
t_slider = st.sidebar.slider("Select Time (s)", 0, int(df.time.max()), 0, 1, key="time_slider")
play = st.sidebar.button("Play Live Simulation")
update_interval = st.sidebar.number_input("Simulation Speed (seconds per frame)", min_value=0.1, max_value=5.0, value=0.75, step=0.1)

# Layout: two side-by-side panels for 3D and 2D views; summary and event logs are below.
col_3d, col_2d = st.columns(2)
col_summary = st.container()

with col_3d:
    st.subheader("3D View")
    radar_placeholder_3d = st.empty()
with col_2d:
    st.subheader("2D Birds‑eye View")
    radar_placeholder_2d = st.empty()
with col_summary:
    st.subheader("Battlefield Summary")
    summary_placeholder = st.empty()
    st.subheader("Event Log")
    event_placeholder = st.empty()

def generate_summary(frame3d, drone_info):
    summary_lines = []
    summary_lines.append("**Current Drone Counts:**")
    for drone_type, count in drone_info.items():
        summary_lines.append(f"- {drone_type}: {count}")
    
    summary_lines.append("\n**Nearest Drone per Unit:**")
    for name, pos in infantry_positions.items():
        min_dist = float("inf")
        closest_drone = None
        for _, row in frame3d.iterrows():
            d = distance_3d(pos, (row.x, row.y, row.z))
            if d < min_dist:
                min_dist = d
                closest_drone = f"{row['type']} ({row['id']})"
        if closest_drone:
            summary_lines.append(f"- {name}: {closest_drone} @ {min_dist:.1f} m")
        else:
            summary_lines.append(f"- {name}: No drone detected")
    return "\n".join(summary_lines)

#######################################
# 6. Simulation / Slider Update Handling
#######################################
if play:
    max_time = int(df.time.max())
    for t in range(t_slider, max_time + 1):
        fig3d, frame3d, drone_info_3d, events3d = plot_radar_frame_3d(t)
        fig2d, frame2d, drone_info_2d, events2d = plot_radar_frame_2d(t)
        
        # Update the side-by-side views.
        radar_placeholder_3d.pyplot(fig3d)
        radar_placeholder_2d.pyplot(fig2d)
        # For summary, using the 3D frame (as overall reference) and merging drone counts.
        all_drone_info = {}
        for key in set(list(drone_info_3d.keys()) + list(drone_info_2d.keys())):
            all_drone_info[key] = drone_info_3d.get(key, 0) + drone_info_2d.get(key, 0)
        summary_placeholder.markdown(generate_summary(frame3d, all_drone_info))
        
        # Merge events from both views.
        events = events3d + events2d
        if events:
            event_placeholder.markdown("\n".join([f"- {e}" for e in events]))
        else:
            event_placeholder.markdown("No new alerts at this time.")
            
        time.sleep(update_interval)
    st.session_state["time_slider"] = max_time
else:
    fig3d, frame3d, drone_info_3d, events3d = plot_radar_frame_3d(t_slider)
    fig2d, frame2d, drone_info_2d, events2d = plot_radar_frame_2d(t_slider)
    radar_placeholder_3d.pyplot(fig3d)
    radar_placeholder_2d.pyplot(fig2d)
    all_drone_info = {}
    for key in set(list(drone_info_3d.keys()) + list(drone_info_2d.keys())):
        all_drone_info[key] = drone_info_3d.get(key, 0) + drone_info_2d.get(key, 0)
    summary_placeholder.markdown(generate_summary(frame3d, all_drone_info))
    events = events3d + events2d
    if events:
        event_placeholder.markdown("\n".join([f"- {e}" for e in events]))
    else:
        event_placeholder.markdown("No critical events at this time.")
