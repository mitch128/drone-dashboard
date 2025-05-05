import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math

#######################################
# 1. Generate More Realistic Dummy Drone Data
#######################################
def make_dummy_data():
    # Instead of a fixed line-by-line approach, we generate trajectories with some curves and noise.
    np.random.seed(42)  # For reproducibility
    timestamps = np.arange(0, 31)  # 0-30 seconds simulation
    rows = []
    
    # Drone D1: High-threat (Shahed) with curved acceleration and small random deviation
    start_D1 = np.array([-600, -500])
    velocity_D1 = np.array([30, 35])  # base velocity per second
    for t in timestamps:
        # Introduce a slight curvature via sine and cosine functions
        deviation = np.array([15 * np.sin(0.1 * t), 15 * np.cos(0.1 * t)])
        noise = np.random.normal(0, 2, 2)  # small noise
        pos = start_D1 + velocity_D1 * t + deviation + noise
        rows.append({
            'time': t,
            'id': 'D1',
            'type': 'Shahed',
            'x': pos[0],
            'y': pos[1]
        })

    # Drone D2: Recon drone (DJI Mavic) following a loopy path 
    start_D2 = np.array([200, 400])
    for t in timestamps:
        # Circular trajectory plus forward drift
        angle = 0.2 * t
        radius = 50 + 0.5 * t  # slow outward spiral
        pos = start_D2 + np.array([radius * np.cos(angle), radius * np.sin(angle)])
        noise = np.random.normal(0, 1.5, 2)
        pos = pos + noise
        rows.append({
            'time': t,
            'id': 'D2',
            'type': 'DJI Mavic',
            'x': pos[0],
            'y': pos[1]
        })
    
    # Drone D3: Surveillance drone (Recon) with a steady linear drift and minor oscillation    
    start_D3 = np.array([-300, 600])
    for t in timestamps:
        # Linear movement with oscillation
        base = start_D3 + np.array([10 * t, -12 * t])
        oscillation = np.array([8 * np.sin(0.15 * t), 8 * np.cos(0.15 * t)])
        noise = np.random.normal(0, 1, 2)
        pos = base + oscillation + noise
        rows.append({
            'time': t,
            'id': 'D3',
            'type': 'Recon',
            'x': pos[0],
            'y': pos[1]
        })
    
    df = pd.DataFrame(rows)
    return df

df = make_dummy_data()

#######################################
# 2. Define Friendly Positions (Frontline Units)
#######################################
infantry_positions = {
    "Alpha (1st Battalion HQ)": (0, 0),
    "Bravo (FOB – Forward Operating Base)": (250, 150),
    "Charlie (Observation Post)": (-200, -150)
}

#######################################
# Utility: Euclidean Distance Calculation
#######################################
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

#######################################
# Utility: Compute Projected Impact Zone for High-threat Drone
#######################################
def compute_impact_zone(x, y, vx, vy, seconds=5):
    # Extrapolate position based on current velocity over a few seconds
    proj_x = x + vx * seconds
    proj_y = y + vy * seconds
    # Create an estimated circular impact area with a radius based on uncertainty (e.g., 10% of distance traveled)
    dist = math.sqrt((vx*seconds)**2 + (vy*seconds)**2)
    impact_radius = 0.1 * dist + 20  # add minimum 20m radius
    return (proj_x, proj_y, impact_radius)

#######################################
# 3. Plotting Function for a Single Timeframe
#######################################
def plot_radar_frame(t):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_facecolor('#F7F7F7')
    ax.set_xlim(-700, 700)
    ax.set_ylim(-700, 700)
    ax.set_title(f"Live Drone Tracker – Time = {t}s", fontsize=16)

    # Draw engagement rings around each infantry position (different rings per unit could be drawn, here we do global rings)
    for r in (100, 250, 500):
        circle = plt.Circle((0, 0), r, fill=False, linestyle='--', color='gray', linewidth=1)
        ax.add_patch(circle)
        ax.text(r - 20, -10, f"{r}m", color='gray', fontsize=8)

    # Plot frontline unit positions
    for name, (ix, iy) in infantry_positions.items():
        ax.plot(ix, iy, 'ks', markersize=12)
        ax.text(ix+10, iy+10, name, fontsize=10, color='black')

    # Plot drones from the current time frame and update log events
    frame = df[df.time == t]
    drone_info = {}
    events = []
    for _, row in frame.iterrows():
        drone_type = row['type']
        # Count drone types for summary
        drone_info[drone_type] = drone_info.get(drone_type, 0) + 1
        
        # Set color and marker based on drone type
        if drone_type == 'Shahed':
            color = 'red'
        elif drone_type == 'DJI Mavic':
            color = 'blue'
        else:
            color = 'green'
        
        # Plot the drone position
        ax.plot(row.x, row.y, 'o', color=color, markersize=10, alpha=0.9)
        ax.text(row.x+10, row.y+10, f"{row['id']}\n{drone_type}", fontsize=9, color=color)
        
        # For high-threat drone, compute and plot projected impact zone 
        if drone_type == 'Shahed':
            # Derive an approximate velocity using a forward difference (if t > 0) else assume base velocity.
            if t > 0:
                prev = df[(df.id == row['id']) & (df.time == t-1)]
                if not prev.empty:
                    vx = row.x - prev.iloc[0].x
                    vy = row.y - prev.iloc[0].y
                else:
                    vx, vy = 30, 35  # fallback
            else:
                vx, vy = 30, 35
            
            proj_x, proj_y, impact_radius = compute_impact_zone(row.x, row.y, vx, vy, seconds=5)
            # Draw projected trajectory arrow
            ax.arrow(row.x, row.y, proj_x-row.x, proj_y-row.y,
                     head_width=15, head_length=15, fc='red', ec='red', alpha=0.6, length_includes_head=True)
            # Draw the impact zone as a circle
            impact_circle = plt.Circle((proj_x, proj_y), impact_radius, fill=True, color='red', alpha=0.2)
            ax.add_patch(impact_circle)
            # Annotate projected point
            ax.text(proj_x+10, proj_y+10, "Estimated Impact\nZone", fontsize=8, color='red')
            # Log an alert event
            events.append(f"ALERT: {row['id']} [{drone_type}] projected impact zone centered at ({proj_x:.1f}, {proj_y:.1f}) with radius {impact_radius:.1f}m")

    ax.grid(True)
    return fig, frame, drone_info, events

#######################################
# 4. Streamlit UI – Command Center Dashboard
#######################################
st.set_page_config(page_title="Drone Intelligence Dashboard", layout="wide")
st.title("Drone Intelligence Dashboard")
st.markdown("""
This tool monitors enemy drone activity relative to frontline units.
Use the control panel to replay the simulation or adjust the simulation speed.
""")

# Sidebar Controls
st.sidebar.header("Command Center Controls")
t_slider = st.sidebar.slider("Select Time (s)", 0, int(df.time.max()), 0, 1, key="time_slider")
play = st.sidebar.button("Play Live Simulation")
update_interval = st.sidebar.number_input("Simulation Speed (seconds per frame)", min_value=0.1, max_value=5.0, value=0.75, step=0.1)

# Layout: left for radar display, right for battlefield summary and event log
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
    # Nearest drone (threat) for each frontline unit
    summary_lines.append("\n**Nearest Drone per Unit:**")
    for name, pos in infantry_positions.items():
        min_dist = float("inf")
        closest_drone = None
        for _, row in frame.iterrows():
            d = distance(pos, (row.x, row.y))
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
        fig, frame, drone_info, events = plot_radar_frame(t)
        radar_placeholder.pyplot(fig)
        summary_placeholder.markdown(generate_summary(frame, drone_info))
        # Display event log if there are alerts; otherwise, display a default message.
        if events:
            event_placeholder.markdown("\n".join([f"- {e}" for e in events]))
        else:
            event_placeholder.markdown("No new alerts at this time.")
        time.sleep(update_interval)
    # Update the slider when simulation completes.
    st.session_state["time_slider"] = max_time
else:
    # Display the current frame based on your slider
    fig, frame, drone_info, events = plot_radar_frame(t_slider)
    radar_placeholder.pyplot(fig)
    summary_placeholder.markdown(generate_summary(frame, drone_info))
    if events:
        event_placeholder.markdown("\n".join([f"- {e}" for e in events]))
    else:
        event_placeholder.markdown("No critical events at this time.")
