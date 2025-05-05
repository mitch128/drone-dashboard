import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
from math import sin, cos

# --- Initial Setup ---
st.set_page_config(layout="wide")
st.title("Drone Detection & Tracking Simulation")

NUM_FRAMES = 200
DRONES = ["Shahed", "FPV-1", "FPV-2"]
BASE_LOCATION = (0, 0)

# --- Initialize Session State ---
if "frame" not in st.session_state:
    st.session_state.frame = 0
if "play" not in st.session_state:
    st.session_state.play = False

# --- Trajectory Generator Functions ---
def generate_shahed_trajectory():
    x = np.linspace(-100, 100, NUM_FRAMES)
    y = np.linspace(-100, 100, NUM_FRAMES)
    z = np.linspace(10, 50, NUM_FRAMES)
    return x, y, z

def generate_fpv_trajectory(seed_offset=0):
    np.random.seed(seed_offset)
    t = np.linspace(0, 4 * np.pi, NUM_FRAMES)
    x = 50 * np.cos(t + np.random.uniform(-0.2, 0.2)) + np.random.normal(0, 5, NUM_FRAMES)
    y = 50 * np.sin(t + np.random.uniform(-0.2, 0.2)) + np.random.normal(0, 5, NUM_FRAMES)
    z = 20 + 10 * np.sin(t + np.random.uniform(-0.5, 0.5)) + np.random.normal(0, 3, NUM_FRAMES)
    return x, y, z

# --- Precompute All Trajectories ---
trajectories = {
    "Shahed": generate_shahed_trajectory(),
    "FPV-1": generate_fpv_trajectory(1),
    "FPV-2": generate_fpv_trajectory(2),
}

# --- UI Controls ---
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("▶ Play"):
        st.session_state.play = True
    if st.button("⏸ Pause"):
        st.session_state.play = False
    st.write(f"Frame: {st.session_state.frame+1}/{NUM_FRAMES}")

# --- Generate 2D and 3D Charts ---
def draw_charts(frame):
    fig2d = go.Figure()
    fig3d = go.Figure()

    # Draw base
    base_x, base_y = BASE_LOCATION
    fig2d.add_trace(go.Scatter(x=[base_x], y=[base_y], mode="markers", marker=dict(size=15, color="green"), name="Base"))
    fig3d.add_trace(go.Scatter3d(x=[base_x], y=[base_y], z=[0], mode="markers", marker=dict(size=8, color="green"), name="Base"))

    for drone in DRONES:
        x, y, z = trajectories[drone]
        current_x = x[:frame+1]
        current_y = y[:frame+1]
        current_z = z[:frame+1]

        fig2d.add_trace(go.Scatter(x=current_x, y=current_y, mode="lines+markers", name=drone))
        fig3d.add_trace(go.Scatter3d(x=current_x, y=current_y, z=current_z, mode="lines+markers", name=drone))

    fig2d.update_layout(title="2D Drone Trajectories", xaxis_title="X", yaxis_title="Y", height=500)
    fig3d.update_layout(title="3D Drone Trajectories", scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Altitude"), height=500)

    return fig2d, fig3d

fig2d, fig3d = draw_charts(st.session_state.frame)

# --- Display Charts Side-by-Side ---
colA, colB = st.columns(2)
with colA:
    st.plotly_chart(fig2d, use_container_width=True)
with colB:
    st.plotly_chart(fig3d, use_container_width=True)

# --- Simulate Playback Loop ---
if st.session_state.play:
    time.sleep(0.1)
    st.session_state.frame += 1
    if st.session_state.frame >= NUM_FRAMES:
        st.session_state.frame = NUM_FRAMES - 1
        st.session_state.play = False
    st.experimental_rerun()
