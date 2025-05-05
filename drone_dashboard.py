# drone_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

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
# No need to sort here unless you have out‐of‐order data

# — 2. Define infantry positions —
infantry_positions = {
    "Alpha": (0, 0),
    "Bravo": (200, 100),
    "Charlie": (-150, -100)
}

# — 3. Plotting function —
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
        ax.text(ix+10, iy+10, name)

    # Plot drones
    frame = df[df.time == t]
    for _, row in frame.iterrows():
        color = 'red' if row['type']=='Shahed' else ('blue' if row['type']=='DJI Mavic' else 'green')
        ax.plot(row.x, row.y, 'o', color=color, markersize=12, alpha=0.8)
        ax.text(row.x+10, row.y+10, f"{row.id} ({row.type})", fontsize=9)
        # Projected path for Shahed
        if row['type']=='Shahed':
            ax.arrow(row.x, row.y, 100, 80, head_width=20, head_length=20,
                     fc='red', ec='red', alpha=0.4)
            ax.text(row.x+100, row.y+80, "Impact", color='red', fontsize=8)

    ax.grid(True)
    return fig

# — 4. Streamlit UI —  
st.title("Drone Radar Dashboard")  
t = st.slider("Time (s)", 0, int(df.time.max()), 0, 1)  
fig = plot_radar_frame(t)  
st.pyplot(fig)
