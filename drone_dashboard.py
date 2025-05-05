import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math

# Page setup
st.set_page_config(layout="wide")
st.title("🔍 Real-Time Drone Detection & Tracking Dashboard")

# Simulate sample data
np.random.seed(42)
n_drones = 5
n_steps = 60  # 60 seconds of data
timestamps = pd.date_range(start="2025-05-05 10:00:00", periods=n_steps, freq="S")

def generate_movement(drone_type, center=(0,0), speed=1.5):
    x, y = center
    path = []
    for t in range(n_steps):
        if drone_type == "Shahed":  # linear, fast
            x += speed
        elif drone_type == "Mavic":  # random movement
            x += np.random.normal(0, 1)
            y += np.random.normal(0, 1)
        path.append((x, y))
    return path

drone_types = ["Shahed", "Mavic", "Shahed", "Mavic", "Shahed"]
drone_data = []

for i, drone_type in enumerate(drone_types):
    path = generate_movement(drone_type, center=(np.random.randint(-500, 500), np.random.randint(-500, 500)), speed=np.random.uniform(2, 5))
    for t, (x, y) in enumerate(path):
        drone_data.append({
            "id": i,
            "timestamp": timestamps[t],
            "drone_type": drone_type,
            "x": x,
            "y": y,
            "certainty": np.clip(np.random.normal(0.85, 0.05), 0.7, 0.99)
        })

df = pd.DataFrame(drone_data)

# Infantry positions
infantry_positions = {
    "Trench A": (-300, -300),
    "Trench B": (0, 0),
    "Trench C": (300, 300)
}

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    play = st.button("▶️ Start Simulation")
    stop = st.button("⏹️ Stop")
    show_paths = st.checkbox("Show Drone Paths", value=True)

# Initialize session state for time_slider
if 'time_slider' not in st.session_state:
    st.session_state.time_slider = 0  # Initialize slider

# Prepare radar visualization
def radar_plot(current_time, df, show_paths=False):
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_xlim(-600, 600)
    ax.set_ylim(-600, 600)
    ax.set_facecolor("white")
    ax.set_title(f"Radar View at {current_time.time()}", fontsize=14)

    # Radar range rings
    for radius in [100, 250, 500]:
        circle = plt.Circle((0, 0), radius, color="gray", fill=False, linestyle='--')
        ax.add_patch(circle)
        ax.text(5, radius-10, f"{radius}m", color='gray')

    # Infantry
    for name, (x, y) in infantry_positions.items():
        ax.plot(x, y, "s", color="blue", markersize=10)
        ax.text(x+10, y+10, name, fontsize=10)

    # Drone plotting
    current_drones = df[df['timestamp'] == current_time]
    for drone_id in current_drones['id'].unique():
        drone = current_drones[current_drones['id'] == drone_id].iloc[0]
        x, y = drone["x"], drone["y"]
        color = "red" if drone["drone_type"] == "Shahed" else "orange"
        label = f"ID {drone_id} ({drone['drone_type']})"
        ax.plot(x, y, "o", color=color, markersize=10)
        ax.text(x+5, y+5, label, fontsize=8)

        # Prediction (for Shahed only)
        if drone["drone_type"] == "Shahed":
            direction = np.array([x, y]) - np.array([0, 0])
            norm = np.linalg.norm(direction)
            if norm > 0:
                unit = direction / norm
                predicted = np.array([x, y]) + unit * 200
                ax.plot([x, predicted[0]], [y, predicted[1]], linestyle="--", color="red")
                ax.text(predicted[0], predicted[1], "→ impact", fontsize=8, color="darkred")

        if show_paths:
            history = df[(df["id"] == drone_id) & (df["timestamp"] <= current_time)]
            ax.plot(history["x"], history["y"], linestyle=":", alpha=0.4, linewidth=1)

    return fig

# Summary panel
def summary_panel(current_time):
    sub_df = df[df['timestamp'] == current_time]
    st.metric("📡 Total Active Drones", len(sub_df))
    shaheds = sub_df[sub_df["drone_type"] == "Shahed"]
    mavics = sub_df[sub_df["drone_type"] == "Mavic"]
    st.metric("🚀 One-Way Drones (Shahed)", len(shaheds))
    st.metric("📷 Recon Drones (Mavic)", len(mavics))

    close_to_base = 0
    for _, drone in sub_df.iterrows():
        for pos in infantry_positions.values():
            distance = math.hypot(drone['x'] - pos[0], drone['y'] - pos[1])
            if distance < 250:
                close_to_base += 1
                break
    st.metric("⚠️ Drones <250m From Any Infantry", close_to_base)

# Real-time simulation
if play:
    placeholder = st.empty()
    stats_col = st.sidebar.container()
    for t in timestamps:
        with placeholder.container():
            col1, col2 = st.columns([2,1])
            with col1:
                fig = radar_plot(t, df, show_paths=show_paths)
                st.pyplot(fig)
            with col2:
                with stats_col:
                    st.subheader("📊 Summary")
                    summary_panel(t)
        time.sleep(0.2)
