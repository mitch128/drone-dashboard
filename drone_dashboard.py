import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from math import sqrt
import plotly.graph_objects as go
import time

# --- Constants ---
COLORS = {'Orlan': 'orange', 'Shahed': 'red', 'Zala': 'green'}
infantry = {'Unit A': (150, -200, 0), 'Unit B': (-200, 100, 0)}

# --- Simulated Data ---
np.random.seed(42)
drone_ids = ['D1', 'D2', 'D3']
drone_types = ['Orlan', 'Shahed', 'Zala']
frames = []
for i, (did, dtype) in enumerate(zip(drone_ids, drone_types)):
    for t in range(31):
        frames.append({
            'id': did,
            'type': dtype,
            'time': t,
            'x': np.sin(0.1 * t + i) * 300 + i * 50,
            'y': np.cos(0.1 * t + i) * 300 + i * 50,
            'z': 50 + 20 * i + np.sin(0.2 * t) * 20
        })
df = pd.DataFrame(frames)

# --- Streamlit Page Config ---
st.set_page_config(layout="wide")
st.title("🛡️ Drone Tracker Dashboard")
st.markdown("Simulated radar view for infantry teams to monitor aerial drone threats in real time.")

# --- Session State Setup ---
if "simulation_time" not in st.session_state:
    st.session_state.simulation_time = 0
if "playing" not in st.session_state:
    st.session_state.playing = False

# --- Layout: Controls Section ---
col_controls, col_charts = st.columns([1, 3])
with col_controls:
    st.write(f"Current time: **{st.session_state.simulation_time}** sec")
    if st.button("Play" if not st.session_state.playing else "Pause"):
        st.session_state.playing = not st.session_state.playing

    slider_time = st.slider(
        "Set Simulation Time (seconds)",
        min_value=0,
        max_value=int(df.time.max()),
        value=st.session_state.simulation_time,
        key="time_slider"
    )
    if slider_time != st.session_state.simulation_time:
        st.session_state.simulation_time = slider_time
        st.session_state.playing = False

# --- Layout: Chart Section (Side by Side) ---
col_chart1, col_chart2 = st.columns(2)
matplotlib_placeholder = col_chart1.empty()
plotly_placeholder = col_chart2.empty()

status_placeholder = st.empty()
alerts_placeholder = st.empty()

# --- Helper Functions ---
def distance(p1, p2):
    return sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def plot_frame(t, three_d=False):
    if not three_d:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_facecolor('#EAEAEA')
        ax.set_xlim(-600, 600)
        ax.set_ylim(-600, 600)
        for r in (100, 250, 500):
            circle = Circle((0, 0), r, fill=False, ls='--', color='gray', lw=1)
            ax.add_patch(circle)
            ax.text(r - 15, 0, f"{r}m", fontsize=8, color='gray')
        ax.set_title(f"Drone Tracker t={t}s", fontsize=14)

        for name, (ix, iy, iz) in infantry.items():
            ax.plot(ix, iy, 'ks', ms=8)
            ax.text(ix + 8, iy + 8, name, fontsize=8)

        for drone_id in df.id.unique():
            path = df[(df.id == drone_id) & (df.time <= t)].sort_values('time')
            if len(path) > 1:
                c = COLORS[path.type.iloc[0]]
                alphas = np.linspace(0.2, 0.6, len(path) - 1)
                for i in range(len(path) - 1):
                    x0, y0 = path.iloc[i][['x', 'y']]
                    x1, y1 = path.iloc[i + 1][['x', 'y']]
                    ax.plot([x0, x1], [y0, y1], ls=':', color=c, alpha=alphas[i])

        now = df[df.time == t]
        counts, events, seen = {}, [], set()
        for _, r in now.iterrows():
            counts[r.type] = counts.get(r.type, 0) + 1
            c = COLORS[r.type]
            ax.plot(r.x, r.y, 'o', c=c, ms=8, label=r.type if r.type not in seen else "")
            ax.text(r.x + 5, r.y + 5, f"{r.id}", fontsize=7)
            seen.add(r.type)
            if r.type == 'Shahed':
                events.append(f"ALERT {r.id} at ({r.x:.0f},{r.y:.0f})")
                ax.arrow(r.x, r.y, 100, 80, head_width=10, head_length=10, fc=c, ec=c, alpha=0.5)

        ax.legend(loc='upper left', fontsize=8)
        return fig, now, counts, events
    else:
        fig = go.Figure()
        now = df[df.time == t]
        hist = df[df.time <= t]
        counts, events = {}, []

        for drone_id, path in hist.groupby("id"):
            c = COLORS[path["type"].iloc[0]]
            fig.add_trace(go.Scatter3d(
                x=path["x"],
                y=path["y"],
                z=path["z"],
                mode="lines",
                name=drone_id,
                line=dict(color=c, width=2)
            ))

        for _, r in now.iterrows():
            counts[r.type] = counts.get(r.type, 0) + 1
            c = COLORS[r.type]
            fig.add_trace(go.Scatter3d(
                x=[r.x],
                y=[r.y],
                z=[r.z],
                mode="markers+text",
                marker=dict(size=6, color=c),
                text=[r.id],
                textposition="top center",
                showlegend=False
            ))
            if r.type == "Shahed":
                events.append(f"ALERT {r.id} at ({r.x:.0f},{r.y:.0f})")

        for name, (ix, iy, iz) in infantry.items():
            fig.add_trace(go.Scatter3d(
                x=[ix],
                y=[iy],
                z=[iz],
                mode="markers+text",
                marker=dict(symbol="square", size=6, color="black"),
                text=[name],
                textposition="top center",
                showlegend=False
            ))

        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-600, 600]),
                yaxis=dict(range=[-600, 600]),
                zaxis=dict(range=[0, 500])
            ),
            height=600,
            margin=dict(l=0, r=0, t=30, b=0),
            title=f"Drone Tracker t={t}s"
        )
        return fig, now, counts, events

# --- Render Function ---
def render(t):
    fig2d, frame2d, counts, events = plot_frame(t, three_d=False)
    matplotlib_placeholder.pyplot(fig2d)

    fig3d, *_ = plot_frame(t, three_d=True)
    plotly_placeholder.plotly_chart(fig3d, use_container_width=True)

    info_lines = ["**Counts:**"] + [f"- {k}: {v}" for k, v in counts.items()] + ["\n**Closest:**"]
    for name, pos in infantry.items():
        dmin, nearest = float("inf"), "—"
        for _, row in frame2d.iterrows():
            d = distance((row.x, row.y, row.z), pos)
            if d < dmin:
                dmin, nearest = d, row.id
        info_lines.append(f"- {name}: {nearest} @ {dmin:.1f}m")

    status_placeholder.markdown("\n".join(info_lines))
    alerts_placeholder.markdown("\n".join(f"- {msg}" for msg in events) or "No alerts.")

# --- Main Simulation Logic ---
def main():
    current_time = st.session_state.simulation_time
    render(current_time)

    # Auto-play logic with protection
    if st.session_state.playing:
        if current_time < df.time.max():
            st.session_state.simulation_time += 1
            st.experimental_rerun()
        else:
            st.session_state.playing = False
            st.experimental_rerun()

if __name__ == "__main__":
    main()
