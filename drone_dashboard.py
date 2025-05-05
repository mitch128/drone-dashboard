import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# --- Page Setup ---
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

# --- Trajectory Generators ---
def generate_shahed_trajectory():
    x = np.linspace(-100, 100, NUM_FRAMES)
    y = np.linspace(-100, 100, NUM_FRAMES)
    z = np.linspace(20, 50, NUM_FRAMES)
    return x, y, z

def generate_fpv_trajectory(seed_offset=0):
    np.random.seed(seed_offset)
    t = np.linspace(0, 4 * np.pi, NUM_FRAMES)
    x = 50 * np.cos(t) + np.random.normal(0, 5, NUM_FRAMES)
    y = 50 * np.sin(t) + np.random.normal(0, 5, NUM_FRAMES)
    z = 30 + 10 * np.sin(t + np.random.uniform(-0.5, 0.5)) + np.random.normal(0, 2, NUM_FRAMES)
    return x, y, z

# --- Precompute All Trajectories ---
trajectories = {
    "Shahed": generate_shahed_trajectory(),
    "FPV-1": generate_fpv_trajectory(1),
    "FPV-2": generate_fpv_trajectory(2),
}

# --- UI Controls ---
left, right = st.columns([1, 5])
with left:
    st.write("### Controls")
    if st.button("▶ Play"):
        st.session_state.play = True
    if st.button("⏸ Pause"):
        st.session_state.play = False
    st.write(f"Frame: {st.session_state.frame + 1} / {NUM_FRAMES}")

# --- Plot Function ---
def draw_charts(frame):
    fig2d = go.Figure()
    fig3d = go.Figure()

    # Add base to both charts
    bx, by = BASE_LOCATION
    fig2d.add_trace(go.Scatter(x=[bx], y=[by], mode="markers", marker=dict(size=12, color="green"), name="Base"))
    fig3d.add_trace(go.Scatter3d(x=[bx], y=[by], z=[0], mode="markers", marker=dict(size=8, color="green"), name="Base"))

    # Add drone trajectories
    for drone in DRONES:
        x, y, z = trajectories[drone]
        fx, fy, fz = x[:frame+1], y[:frame+1], z[:frame+1]
        fig2d.add_trace(go.Scatter(x=fx, y=fy, mode="lines+markers", name=drone))
        fig3d.add_trace(go.Scatter3d(x=fx, y=fy, z=fz, mode="lines+markers", name=drone))

    fig2d.update_layout(title="2D Trajectories", xaxis_title="X", yaxis_title="Y", height=500)
    fig3d.update_layout(title="3D Trajectories", scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Altitude"), height=500)

    return fig2d, fig3d

# --- Chart Placeholders ---
placeholder2d, placeholder3d = st.columns(2)
plot2d = placeholder2d.empty()
plot3d = placeholder3d.empty()

# --- Draw initial plots ---
fig2d, fig3d = draw_charts(st.session_state.frame)
plot2d.plotly_chart(fig2d, use_container_width=True)
plot3d.plotly_chart(fig3d, use_container_width=True)

# --- Playback Loop ---
if st.session_state.play:
    for _ in range(NUM_FRAMES - st.session_state.frame):
        time.sleep(0.1)
        st.session_state.frame += 1
        fig2d, fig3d = draw_charts(st.session_state.frame)
        plot2d.plotly_chart(fig2d, use_container_width=True)
        plot3d.plotly_chart(fig3d, use_container_width=True)
        if not st.session_state.play:
            break
