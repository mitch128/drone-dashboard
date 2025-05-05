import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math

# — 1. Generate dummy drone data —
def make_dummy_data():
    # Simulate timestamps 0–10
    timestamps = np.arange(0, 11)
    rows = []
    for t in timestamps:
        # One‐way attack drone
        rows.append({'time': t, 'id': 'D1', 'type': 'Shahed', 
                     'x': -400 + 50 * t, 'y': -300 + 40 * t})
        # Hover‐capable drone
        rows.append({'time': t, 'id': 'D2', 'type': 'DJI Mavic', 
                     'x': 100 + 10 * np.sin(t), 'y': 150 + 10 * np.cos(t)})
        # Recon drone
        rows.append({'time': t, 'id': 'D3', 'type': 'Recon', 
                     'x': -200 + 5 * t, 'y': 300 - 2 * t})
    df = pd.DataFrame(rows)
    return df

df = make_dummy_data()

# — 2. Define infantry positions —
infantry_positions = {
    "Alpha": (0, 0),
    "Bravo": (200, 100),
    "Charlie": (-150, -100)
}

# — Utility: Calculate Euclidean distance between two points —
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# — 3. Plotting function for one frame—
def plot_radar_frame(t):
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_facecolor('white')
    ax.set_xlim(-600, 600)
    ax.set_ylim(-600, 600)
    ax.set_title(f"Drone Radar – t = {t}s")

    # Draw range rings at 100, 250, 500 m
    for r in (100, 250, 500):
        circle = plt.Circle((0,0), r, fill=False, linestyle='--', color='gray')
        ax.add_patch(circle)
        ax.text(r, 0, f"{r}m", color='gray')

    # Plot infantry
    for name, (ix, iy) in infantry_positions.items():
        ax.plot(ix, iy, 'ks', markersize=10)
        ax.text(ix+10, iy+10, name, fontsize=9, color='black')

    # Plot drones and collect summary info
    frame = df[df.time == t]
    drone_info = {}  # key: type, value: count
    for _, row in frame.iterrows():
        # Update count
        drone_info[row['type']] = drone_info.get(row['type'], 0) + 1

        # Choose color based on drone type
        if row['type'] == 'Shahed':
            color = 'red'
        elif row['type'] == 'DJI Mavic':
            color = 'blue'
        else:
            color = 'green'

        ax.plot(row.x, row.y, 'o', color=color, markersize=12, alpha=0.8)
        ax.text(row.x+10, row.y+10, f"{row.id} ({row['type']})", fontsize=9)
        # Draw projected path (for the Shahed drone)
        if row['type']=='Shahed':
            ax.arrow(row.x, row.y, 100, 80, head_width=20, head_length=20,
                     fc='red', ec='red', alpha=0.4)
            ax.text(row.x+100, row.y+80, "Impact", color='red', fontsize=8)

    ax.grid(True)
    return fig, frame, drone_info

# — 4. Streamlit UI Setup —  
st.title("Drone Radar Dashboard")

# Create two columns: one for control buttons and slider, one for summary stats
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Controls")
    # Slider to manually select time
    t_slider = st.slider("Time (s)", 0, int(df.time.max()), 0, 1, key='time_slider')
    play = st.button("Play Real Time")

# Placeholder for the plot
plot_placeholder = st.empty()

# Placeholder for summary stats
summary_placeholder = st.empty()

# Function to generate summary stats based on current frame and infantry positions
def generate_summary(frame, drone_info):
    summary_lines = []
    summary_lines.append("### Summary Statistics")
    
    # Drone counts by type
    summary_lines.append("**Drone Counts:**")
    for drone_type, count in drone_info.items():
        summary_lines.append(f"- {drone_type}: {count}")
    
    # Nearest drone distances for each infantry unit
    summary_lines.append("\n**Nearest Drone per Infantry Unit:**")
    for name, pos in infantry_positions.items():
        min_dist = float("inf")
        which_drone = None
        for _, row in frame.iterrows():
            d = distance(pos, (row.x, row.y))
            if d < min_dist:
                min_dist = d
                which_drone = row['id']
        if which_drone:
            summary_lines.append(f"- {name}: Drone {which_drone} at {min_dist:.1f} m")
        else:
            summary_lines.append(f"- {name}: No drone detected")
    return "\n".join(summary_lines)

# Real-time animation or manual slider update handling:
if play:
    # Disable slider when playing animation
    max_time = int(df.time.max())
    for t in range(t_slider, max_time+1):
        fig, frame, drone_info = plot_radar_frame(t)
        plot_placeholder.pyplot(fig)
        summary_placeholder.markdown(generate_summary(frame, drone_info))
        time.sleep(1)  # Pause for 1 second for demonstration; adjust timing as needed
    # Reset slider value to the last frame after finishing play
    st.session_state.time_slider = max_time
else:
    # When not playing, simply update based on slider
    fig, frame, drone_info = plot_radar_frame(t_slider)
    plot_placeholder.pyplot(fig)
    summary_placeholder.markdown(generate_summary(frame, drone_info))
