import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import math

#######################################
# 1. Generate Realistic Dummy Drone Data
#######################################
def make_dummy_data():
    # Simulated timeline of 0-10 seconds
    timestamps = np.arange(0, 11)
    rows = []
    for t in timestamps:
        # High-threat attack drone (Shahed)
        rows.append({
            'time': t,
            'id': 'D1',
            'type': 'Shahed',
            'x': -400 + 50 * t,
            'y': -300 + 40 * t,
            'z': 100
        })
        # Recon drone (DJI Mavic)
        rows.append({
            'time': t,
            'id': 'D2',
            'type': 'DJI Mavic',
            'x': 100 + 10 * np.sin(t),
            'y': 150 + 10 * np.cos(t),
            'z': 50
        })
        # Surveillance drone (Recon)
        rows.append({
            'time': t,
            'id': 'D3',
            'type': 'Recon',
            'x': -200 + 5 * t,
            'y': 300 - 2 * t,
            'z': 200
        })
    df = pd.DataFrame(rows)
    return df

df = make_dummy_data()

#######################################
# 2. Define Friendly Positions (Frontline Units)
#######################################
infantry_positions = {
    "Alpha (1st Battalion HQ)": (0, 0, 0),
    "Bravo (FOB – Forward Operating Base)": (200, 100, 0),
    "Charlie (Observation Post)": (-150, -100, 0)
}

#######################################
# Utility: Euclidean Distance Calculation
#######################################
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

#######################################
# 3. Plotting Function for a Single Timeframe
#######################################
def plot_radar_frame(t):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_facecolor('#EAEAEA')
    ax.set_xlim(-600, 600)
    ax.set_ylim(-600, 600)
    ax.set_title(f"Live Drone Tracker – t = {t}s", fontsize=16)

    # Draw engagement rings (100m, 250m, 500m)
    for r in (100, 250, 500):
        circle = plt.Circle((0, 0), r, fill=False, linestyle='--', color='gray', linewidth=1)
        ax.add_patch(circle)
        ax.text(r - 20, 0, f"{r}m", color='gray', fontsize=8)

    # Plot frontline unit positions
    for name, (ix, iy, iz) in infantry_positions.items():
        ax.plot(ix, iy, 'ks', markersize=12)
        ax.text(ix + 10, iy + 10, name, fontsize=10, color='black')

    # Plot drones and collect events for log display
    frame = df[df.time == t]
    drone_info = {}
    events = []
    for _, row in frame.iterrows():
        # Count drone types
        drone_info[row['type']] = drone_info.get(row['type'], 0) + 1
        # Set color for drone type
        if row['type'] == 'Shahed':  # High-threat attack drone
            color = 'red'
            # Log event for high-threat drone detection
            events.append(f"ALERT: {row['type']} ({row['id']}) detected at ({row['x']}, {row['y']})")
        elif row['type'] == 'DJI Mavic':  # Recon drone
            color = 'blue'
        else:  # Surveillance drone
            color = 'green'

        # Plot drone position
        ax.plot(row.x, row.y, 'o', color=color, markersize=12, alpha=0.9)
        ax.text(row.x + 10, row.y + 10, f"{row.id}\n({row['type']})", fontsize=9, color=color)
        
        # For "Shahed", show projected trajectory and potential impact zone
        if row['type'] == 'Shahed':
                        ax.arrow(row.x, row.y, 100, 80, head_width=20, head_length=20,
                     fc='red', ec='red', alpha=0.5)
            ax.text(row.x + 100, row.y + 80, "THREAT AREA", color='red', fontsize=10)

        # Plot trajectory dotted lines
        previous_row = df[(df['id'] == row['id']) & (df['time'] < t)].sort_values(by='time', ascending=False).head(1)
        if not previous_row.empty:
            previous_row = previous_row.iloc[0]
            ax.plot([previous_row['x'], row['x']], [previous_row['y'], row['y']], linestyle='--', color=color, alpha=0.5)

    ax.grid(True)
    return fig, frame, drone_info, events

def plot_3d_frame(t):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Live Drone Tracker – t = {t}s", fontsize=16)

    # Plot frontline unit positions
    for name, (ix, iy, iz) in infantry_positions.items():
        ax.scatter(ix, iy, iz, c='k', marker='s', s=100)
        ax.text(ix, iy, iz, name, fontsize=10, color='black')

    # Plot drones and collect events for log display
    frame = df[df.time == t]
    drone_info = {}
    events = []
    for _, row in frame.iterrows():
        # Count drone types
        drone_info[row['type']] = drone_info.get(row['type'], 0) + 1
        # Set color for drone type
        if row['type'] == 'Shahed':  # High-threat attack drone
            color = 'red'
            # Log event for high-threat drone detection
            events.append(f"ALERT: {row['type']} ({row['id']}) detected at ({row['x']}, {row['y']})")
        elif row['type'] == 'DJI Mavic':  # Recon drone
            color = 'blue'
        else:  # Surveillance drone
            color = 'green'

        # Plot drone position
        ax.scatter(row.x, row.y, row.z, c=color, marker='o', s=100)
        ax.text(row.x, row.y, row.z, f"{row.id}\n({row['type']})", fontsize=9, color=color)

        # For "Shahed", show projected trajectory and potential impact zone
        if row['type'] == 'Shahed':
            ax.quiver(row.x, row.y, row.z, 100, 80, 0, color='red', alpha=0.5)
            ax.text(row.x + 100, row.y + 80, row.z, "THREAT AREA", color='red', fontsize=10)

        # Plot trajectory dotted lines
        previous_row = df[(df['id'] == row['id']) & (df['time'] < t)].sort_values(by='time', ascending=False).head(1)
        if not previous_row.empty:
            previous_row = previous_row.iloc[0]
            ax.plot([previous_row['x'], row['x']], [previous_row['y'], row['y']], [previous_row['z'], row['z']], linestyle='--', color=color, alpha=0.5)

    ax.set_xlim(-600, 600)
    ax.set_ylim(-600, 600)
    ax.set_zlim(0, 500)
    return fig, frame, drone_info, events

#######################################
# 4. Streamlit UI – Command Center Dashboard
#######################################
st.set_page_config(page_title="Drone Intelligence Dashboard", layout="wide")
st.title("Drone Intelligence Dashboard")
st.markdown("""
This tool monitors enemy drone activity relative to Ukrainian frontline units.
Use the control panel to replay the simulation or adjust the animation speed as needed.
""")

# Sidebar Controls
st.sidebar.header("Command Center Controls")
t_slider = st.sidebar.slider("Select Time (s)", 0, int(df.time.max()), 0, 1, key="time_slider")
play = st.sidebar.button("Play Live Simulation")
update_interval = st.sidebar.number_input("Simulation Speed (seconds per frame)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

# Main panel layout: left for radar display, right for summary and events
col_radar, col_summary = st.columns([2, 1])

with col_radar:
    radar_placeholder = st.empty()
    radar_3d_placeholder = st.empty()
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
    # Nearest threat per frontline unit
    summary_lines.append("\n**Nearest Drone per Unit:**")
    for name, pos in infantry_positions.items():
        min_dist = float("inf")
        closest_drone = None
        for _, row in frame.iterrows():
            d = distance((row['x'], row['y'], row['z']), pos)
            if d < min_dist:
                min_dist = d
                closest_drone = f"{row['type']} ({row['id']})"
        if closest_drone:
            summary_lines.append(f"- {name}: {closest_drone} at {min_dist:.1f}m")
        else:
            summary_lines.append(f"- {name}: No drone detected")
    return "\n".join(summary_lines)

#######################################
# 5. Real-time Simulation / Slider Update Handling
#######################################
if play:
    max_time = int(df.time.max())
    # Animate from the current slider value to the end of the simulation timeline.
    for t in range(t_slider, max_time + 1):
        fig, frame, drone_info, events = plot_radar_frame(t)
        radar_placeholder.pyplot(fig)
        fig_3d, frame_3d, drone_info_3d, events_3d = plot_3d_frame(t)
        radar_3d_placeholder.pyplot(fig_3d)
        summary_placeholder.markdown(generate_summary(frame, drone_info))
        # Display event log if there are any alerts; if not, say "No new events"
        if events:
            event_placeholder.markdown("\n".join([f"- {e}" for e in events]))
        else:
            event_placeholder.markdown("No critical events at this time")
        time.sleep(update_interval)
    # Update slider value for next iteration using dictionary syntax
    st.session_state["time_slider"] = max_time
else:
    # Display the current frame based on the slider value
    fig, frame, drone_info, events = plot_radar_frame(t_slider)
    radar_placeholder.pyplot(fig)
    fig_3d, frame_3d, drone_info_3d, events_3d = plot_3d_frame(t_slider)
    radar_3d_placeholder.pyplot(fig_3d)
    summary_placeholder.markdown(generate_summary(frame, drone_info))
    if events:
        event_placeholder.markdown("\n".join([f"- {e}" for e in events]))
    else:
        event_placeholder.markdown("No critical events at this time")


