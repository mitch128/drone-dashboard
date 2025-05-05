import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math

# -------------------------------
# 1. Generate realistic dummy drone data
# -------------------------------
def make_dummy_data():
    # Simulate 0-10 seconds of data
    timestamps = np.arange(0, 11)
    rows = []
    for t in timestamps:
        # Attack drone ("Shahed")
        rows.append({
            'time': t,
            'id': 'D1',
            'type': 'Shahed', 
            'x': -400 + 50 * t,
            'y': -300 + 40 * t})
        # Recon drone ("DJI Mavic")
        rows.append({
            'time': t,
            'id': 'D2',
            'type': 'DJI Mavic', 
            'x': 100 + 10 * np.sin(t),
            'y': 150 + 10 * np.cos(t)})
        # Surveillance drone ("Recon")
        rows.append({
            'time': t,
            'id': 'D3',
            'type': 'Recon', 
            'x': -200 + 5 * t,
            'y': 300 - 2 * t})
    df = pd.DataFrame(rows)
    return df

df = make_dummy_data()

# -------------------------------
# 2. Define friendly positions (Infantry Units)
# For a realistic battlefield scenario, these represent Ukrainian frontline units.
# -------------------------------
infantry_positions = {
    "Alpha (1st Battalion HQ)": (0, 0),
    "Bravo (Forward Operating Base)": (200, 100),
    "Charlie (Observation Post)": (-150, -100)
}

# -------------------------------
# Utility: Euclidean distance between two positions
# -------------------------------
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# -------------------------------
# 3. Plotting function for a single timeframe
# -------------------------------
def plot_radar_frame(t):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_facecolor('#f0f0f0')
    ax.set_xlim(-600, 600)
    ax.set_ylim(-600, 600)
    ax.set_title(f"Live Drone Tracker – Time: {t}s", fontsize=16)

    # Draw range rings at 100, 250, and 500 meters (typical engagement or observation ranges)
    for r in (100, 250, 500):
        circle = plt.Circle((0, 0), r, fill=False, linestyle='--', color='gray')
        ax.add_patch(circle)
        ax.text(r, 0, f"{r}m", color='gray', fontsize=10)

    # Plot friendly unit positions with military-specific labels
    for name, (ix, iy) in infantry_positions.items():
        ax.plot(ix, iy, 'ks', markersize=12)
        ax.text(ix+10, iy+10, name, fontsize=10, color='black')

    # Plot drones and collect summary information
    frame = df[df.time == t]
    drone_info = {}
    for _, row in frame.iterrows():
        # Count each drone type
        drone_info[row['type']] = drone_info.get(row['type'], 0) + 1

        # Select colors based on drone type
        if row['type'] == 'Shahed':    # High threat attack drone – red
            color = 'red'
        elif row['type'] == 'DJI Mavic':  # Recon drone – blue
            color = 'blue'
        else:                           # Surveillance drone – green
            color = 'green'

        ax.plot(row.x, row.y, 'o', color=color, markersize=12, alpha=0.9)
        ax.text(row.x+10, row.y+10, f"{row.id}\n({row['type']})", fontsize=9, color=color)

        # For the high-threat Shahed drone, indicate its projected path and potential impact point.
        if row['type'] == 'Shahed':
            ax.arrow(row.x, row.y, 100, 80, head_width=20, head_length=20,
                     fc='red', ec='red', alpha=0.5)
            ax.text(row.x+100, row.y+80, "Threat", color='red', fontsize=10)

    ax.grid(True)
    return fig, frame, drone_info

# -------------------------------
# 4. Streamlit UI for Deployment on the Frontline
# -------------------------------
st.title("Live Drone Tracker Dashboard")
st.markdown("Monitoring enemy drone activity relative to Ukrainian frontline units")

# Layout: Controls on left, status/stats on right
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Control Panel")
    t_slider = st.slider("Select Time (s)", 0, int(df.time.max()), 0, 1, key='time_slider')
    play = st.button("Play Live Simulation")
    
# Placeholders for the radar plot and summary information
plot_placeholder = st.empty()
summary_placeholder = st.empty()

# Generate summary statistics based on the current frame and infantry positions
def generate_summary(frame, drone_info):
    summary_lines = []
    summary_lines.append("### Battlefield Summary")
    
    # Drone counts per type
    summary_lines.append("**Active Drone Counts:**")
    for drone_type, count in drone_info.items():
        summary_lines.append(f"- {drone_type}: {count}")
    
    # For each friendly unit, calculate the nearest detected drone
    summary_lines.append("\n**Nearest Drone to Each Unit:**")
    for name, pos in infantry_positions.items():
        min_dist = float("inf")
        threat_id = None
        threat_type = None
        for _, row in frame.iterrows():
            d = distance(pos, (row.x, row.y))
            if d < min_dist:
                min_dist = d
                threat_id = row['id']
                threat_type = row['type']
        if threat_id:
            summary_lines.append(f"- {name}: {threat_type} ({threat_id}) at {min_dist:.1f}m")
        else:
            summary_lines.append(f"- {name}: No threat detected")
    return "\n".join(summary_lines)

# -------------------------------
# Real-time simulation / slider update handling
# -------------------------------
if play:
    # When "Play Live Simulation" is pressed, animate from the selected time to the end.
    max_time = int(df.time.max())
    for t in range(t_slider, max_time+1):
        fig, frame, drone_info = plot_radar_frame(t)
        plot_placeholder.pyplot(fig)
        summary_placeholder.markdown(generate_summary(frame, drone_info))
        time.sleep(1)  # Pause for a 1-second update interval (adjust as required)
    st.session_state["time_slider"] = max_time  # Using dictionary syntax to update session_state
else:
    # When not playing, simply update based on the slider value.
    fig, frame, drone_info = plot_radar_frame(t_slider)
    plot_placeholder.pyplot(fig)
    summary_placeholder.markdown(generate_summary(frame, drone_info))
