import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(layout="wide")
st.title("3D Drone Tracker Simulation")

# ----- Simulated Data -----
np.random.seed(42)
NUM_DRONES = 10
NUM_TIMESTEPS = 100

# Generate fake 3D drone trajectories over time
drone_ids = [f"Drone {i+1}" for i in range(NUM_DRONES)]
timesteps = list(range(NUM_TIMESTEPS))
data = []

for t in timesteps:
    for drone in drone_ids:
        x = 500 * np.sin(2 * np.pi * (t + hash(drone) % 100) / NUM_TIMESTEPS) + np.random.randn() * 30
        y = 500 * np.cos(2 * np.pi * (t + hash(drone) % 100) / NUM_TIMESTEPS) + np.random.randn() * 30
        z = np.abs(100 + 50 * np.sin(t / 10.0) + np.random.randn() * 5)
        data.append({"Time": t, "DroneID": drone, "X": x, "Y": y, "Z": z})

df = pd.DataFrame(data)

# ----- Sidebar Controls -----
st.sidebar.header("Controls")
step = st.sidebar.slider("Time step", 0, NUM_TIMESTEPS - 1, 0)
auto_play = st.sidebar.checkbox("Auto Play")

# ----- 3D Plot -----
def plot_drones_3d(df_timestep):
    fig = go.Figure()

    for drone_id, group in df_timestep.groupby("DroneID"):
        fig.add_trace(go.Scatter3d(
            x=group["X"],
            y=group["Y"],
            z=group["Z"],
            mode="markers+lines",
            marker=dict(size=5),
            name=drone_id
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Altitude (Z)"
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )
    return fig

# ----- Statistics -----
def show_statistics(df_timestep):
    avg_altitude = df_timestep["Z"].mean()
    active_drones = df_timestep["DroneID"].nunique()
    st.metric("Average Altitude", f"{avg_altitude:.1f} m")
    st.metric("Active Drones", active_drones)

# ----- Main View -----
if auto_play:
    play_speed = st.sidebar.slider("Playback speed (s per frame)", 0.01, 1.0, 0.1, step=0.01)
    placeholder_plot = st.empty()
    placeholder_stats = st.empty()

    for t in range(step, NUM_TIMESTEPS):
        df_t = df[df["Time"] == t]
        with placeholder_plot.container():
            st.plotly_chart(plot_drones_3d(df_t), use_container_width=True)
        with placeholder_stats.container():
            show_statistics(df_t)
        time.sleep(play_speed)
else:
    df_step = df[df["Time"] == step]
    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(plot_drones_3d(df_step), use_container_width=True)
    with col2:
        show_statistics(df_step)
