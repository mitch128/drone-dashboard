import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
import time
import math

#######################################
# 1. Generate More Realistic Dummy Drone Data in 3D
#######################################
def make_dummy_data():
    # We'll simulate 3 drones (each with id and type) for a 30sec simulation in 3D (x, y, z).
    np.random.seed(42)  # reproducibility
    timestamps = np.arange(0, 31)  # 0 to 30 seconds
    rows = []
    
    # For each drone we will compute the position and also derive a "velocity" vector from the algorithm below.
    # In the data, for t==0 leave velocity as np.nan (later you can compute approximate velocity differences)
    
    # Drone D1: High-threat (Shahed) with a curved 3D trajectory. It descends as it approaches.
    start_D1 = np.array([-600, -500, 200])   # starting coordinates (x,y,z)
    base_velocity_D1 = np.array([30, 35, -4])  # per second movement in x, y, z
    for t in timestamps:
        # Introduce slight curvature using sinusoidal functions for lateral movement plus noise.
        deviation = np.array([15 * np.sin(0.1*t), 15 * np.cos(0.1*t), 5 * np.sin(0.05*t)])
        noise = np.random.normal(0, 2, 3)
        pos = start_D1 + base_velocity_D1 * t + deviation + noise
        # For velocity, if t == 0 then mark as NaN; else we'll compute a difference with previous second.
        vel = np.array([np.nan, np.nan, np.nan]) if t == 0 else None
        rows.append({
            'time': t,
            'id': 'D1',
            'type': 'Shahed',
            'x': pos[0],
            'y': pos[1],
            'z': pos[2],
            'velocity_x': vel[0] if t==0 else None,
            'velocity_y': vel[1] if t==0 else None,
            'velocity_z': vel[2] if t==0 else None
        })
    
    # Drone D2: Recon drone (DJI Mavic) following a looping spiral in 3D
    start_D2 = np.array([200, 400, 100])
    # We'll have a steady upward drift in z.
    for t in timestamps:
        angle = 0.2*t
        radius = 50 + 0.5*t  # outward spiral
        pos = start_D2 + np.array([radius*np.cos(angle), radius*np.sin(angle), 2*t])
        noise = np.random.normal(0, 1.5, 3)
        pos = pos + noise
        rows.append({
            'time': t,
            'id': 'D2',
            'type': 'DJI Mavic',
            'x': pos[0],
            'y': pos[1],
            'z': pos[2],
            'velocity_x': np.nan if t==0 else None,
            'velocity_y': np.nan if t==0 else None,
            'velocity_z': np.nan if t==0 else None
        })

    # Drone D3: Surveillance drone (Recon) with a steady drift while oscillating in z
    start_D3 = np.array([-300, 600, 150])
    for t in timestamps:
        base = start_D3 + np.array([10*t, -12*t, 1*t])
        oscillation = np.array([8*np.sin(0.15*t), 8*np.cos(0.15*t), 3*np.sin(0.2*t)])
        noise = np.random.normal(0, 1, 3)
        pos = base + oscillation + noise
        rows.append({
            'time': t,
            'id': 'D3',
            'type': 'Recon',
            'x': pos[0],
            'y': pos[1],
            'z': pos[2],
            'velocity_x': np.nan if t==0 else None,
            'velocity_y': np.nan if t==0 else None,
            'velocity_z': np.nan if t==0 else None
        })
    
    df = pd.DataFrame(rows)
    
    # Calculate approximate velocity for each drone after t > 0 using backward differences.
    for drone_id in df['id'].unique():
        drone_mask = df['id'] == drone_id
        df.loc[drone_mask, ['velocity_x', 'velocity_y', 'velocity_z']] = df[drone_mask][['x','y','z']].diff().fillna(0)
    
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
    # Extrapolate future position in 3D based on the current velocity
    proj_x = x + vx * seconds
    proj_y = y + vy * seconds
    proj_z = z + vz * seconds
    # Here, we define the impact zone as a sphere with radius proportional to the displacement magnitude and uncertainty.
    displacement = math.sqrt((vx*seconds)**2 + (vy*seconds)**2 + (vz*seconds)**2)
    impact_radius = 0.1 * displacement + 20  # add a minimum uncertainty radius
    return (proj_x, proj_y, proj_z, impact_radius)

#######################################
# 3. 3D Plotting Function for a Single Timeframe
#######################################
def plot_radar_frame_3d(t):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#F7F7F7')
    ax.set_title(f"3D Drone Tracker – Time = {t}s", fontsize=16)
    
    # Set the limits (you can adjust these based on your simulated area)
    ax.set_xlim(-700, 700)
    ax.set_ylim(-700, 700)
    ax.set_zlim(0, 400)
    
    # Draw ground plane (for friendly unit positions)
    xx, yy = np.meshgrid(np.linspace(-700,700,2), np.linspace(-700,700,2))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color='lightgray', alpha=0.3)
    
    # Plot friendly unit positions
    for name, (ix, iy, iz) in infantry_positions.items():
        ax.scatter(ix, iy, iz, c='k', marker='^', s=100)
        ax.text(ix+10, iy+10, iz+10, name, fontsize=10, color='black')

    # Plot drones and collect events for log display
    frame = df[df.time == t]
    drone_info = {}
    events = []
    for _, row in frame.iterrows():
        drone_type = row['type']
        # Count drone types for summary
        drone_info[drone_type] = drone_info.get(drone_type, 0) + 1
        
        # Use different colors per drone type
        if drone_type == 'Shahed':
            color = 'red'
        elif drone_type == 'DJI Mavic':
            color = 'blue'
        else:
            color = 'green'
        
        # Plot the drone position in 3D space
        ax.scatter(row.x, row.y, row.z, c=color, marker='o', s=80, alpha=0.9)
        # Annotate the drone with id, type, and velocity magnitude
        vel_mag = math.sqrt(row.velocity_x**2 + row.velocity_y**2 + row.velocity_z**2)
        ax.text(row.x+10, row.y+10, row.z+10, f"{row['id']}\n{drone_type}\nV:{vel_mag:.1f}", fontsize=9, color=color)
        
        # For high-threat drone, compute the projected impact zone
        if drone_type == 'Shahed':
            vx, vy, vz = row.velocity_x, row.velocity_y, row.velocity_z
            proj_x, proj_y, proj_z, impact_radius = compute_impact_zone(row.x, row.y, row.z, vx, vy, vz, seconds=5)
            # Draw an arrow representing projected movement vector (using quiver)
            ax.quiver(row.x, row.y, row.z, proj_x-row.x, proj_y-row.y, proj_z-row.z,
                      color='red', length=1, normalize=False, arrow_length_ratio=0.15, alpha=0.6)
            # Draw projected impact zone as a wireframe sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            xs = proj_x + impact_radius * np.outer(np.cos(u), np.sin(v))
            ys = proj_y + impact_radius * np.outer(np.sin(u), np.sin(v))
            zs = proj_z + impact_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_wireframe(xs, ys, zs, color='red', alpha=0.3)
            # Annotate the sphere center
            ax.text(proj_x+10, proj_y+10, proj_z+10, "Impact Zone", fontsize=8, color='red')
            # Log event
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
    # Find the nearest drone for each friendly unit (in 3D)
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
    # Animate from the current slider value to the end of simulation timeline.
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
    # Show the current frame based on slider
    fig, frame, drone_info, events = plot_radar_frame_3d(t_slider)
    radar_placeholder.pyplot(fig)
    summary_placeholder.markdown(generate_summary(frame, drone_info))
    if events:
        event_placeholder.markdown("\n".join([f"- {e}" for e in events]))
    else:
        event_placeholder.markdown("No critical events at this time.")
