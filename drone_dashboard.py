import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time, math
from functools import lru_cache

#######################################
# 1. Generate Realistic Dummy Drone Data
#######################################
@st.cache_data
def make_dummy_data():
    timestamps = np.arange(0, 11)
    rows = []
    for t in timestamps:
        rows += [
            dict(time=t, id='D1', type='Shahed',     x=-400 + 50 * t,          y=-300 + 40 * t,          z=100),
            dict(time=t, id='D2', type='DJI Mavic',  x=100 + 10 * np.sin(t),   y=150 + 10 * np.cos(t),   z=50),
            dict(time=t, id='D3', type='Recon',      x=-200 + 5 * t,           y=300 - 2 * t,            z=200),
        ]
    return pd.DataFrame(rows)

df = make_dummy_data()

#######################################
# 2. Constants & Utilities
#######################################
infantry_positions = {
    "Alpha (1st Battalion HQ)": (0,    0,   0),
    "Bravo (Forward Base)"      : (200,100,   0),
    "Charlie (Observation Post)": (-150,-100,  0),
}

# assign a consistent color per drone type
TYPE_COLORS = {
    'Shahed'     : 'red',
    'DJI Mavic'  : 'blue',
    'Recon'      : 'green',
}

def distance(p1, p2):
    return math.dist(p1, p2)

#######################################
# 3. Plotting Helpers
#######################################
def add_infantry(ax, is3d=False):
    for name, (ix, iy, iz) in infantry_positions.items():
        if is3d:
            ax.scatter(ix, iy, iz, c='k', marker='s', s=100)
            ax.text(ix, iy, iz, name, fontsize=9)
        else:
            ax.plot(ix, iy, 'ks', markersize=12)
            ax.text(ix+10, iy+10, name, fontsize=9)

def add_rings(ax):
    for r in (100, 250, 500):
        circle = plt.Circle((0,0), r, fill=False, linestyle='--', color='gray', lw=1)
        ax.add_patch(circle)
        ax.text(r-20, 0, f"{r}m", color='gray', fontsize=8)

def plot_frame(t, three_d=False):
    # select plotting space
    if three_d:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_facecolor('#EAEAEA')
        ax.set_xlim(-600, 600)
        ax.set_ylim(-600, 600)
        add_rings(ax)

    title = f"Live Drone Tracker – t = {t}s"
    ax.set_title(title, fontsize=16)

    add_infantry(ax, is3d=three_d)

    frame = df[df.time == t]
    drone_counts, events = {}, []
    seen_types = set()

    # Plot each drone, plus full path up to t
    for drone_id, path in df[df.time <= t].groupby('id'):
        dtype = path['type'].iloc[0]
        color = TYPE_COLORS[dtype]
        xs, ys, zs = path['x'].values, path['y'].values, path['z'].values

        # full trajectory dashed
        if three_d:
            ax.plot(xs, ys, zs, linestyle='--', color=color, alpha=0.4)
        else:
            ax.plot(xs, ys, linestyle='--', color=color, alpha=0.4)

    # then plot current positions on top
    for _, row in frame.iterrows():
        drone_counts[row['type']] = drone_counts.get(row['type'], 0) + 1
        color = TYPE_COLORS[row['type']]

        # scatter current pos
        if three_d:
            ax.scatter(row.x, row.y, row.z, c=color, s=100, label=row['type'] if row['type'] not in seen_types else "")
            ax.text(row.x, row.y, row.z, f"{row.id}\n({row['type']})", fontsize=9)
        else:
            ax.plot(row.x, row.y, 'o', c=color, markersize=12, alpha=0.9, label=row['type'] if row['type'] not in seen_types else "")
            ax.text(row.x+10, row.y+10, f"{row.id}\n({row['type']})", fontsize=9)

        seen_types.add(row['type'])

        # threat arrow & log
        if row['type']=='Shahed':
            events.append(f"ALERT: {row['type']} ({row['id']}) at ({row['x']},{row['y']})")
            if three_d:
                ax.quiver(row.x, row.y, row.z, 100, 80, 0, color=color, alpha=0.5)
                ax.text(row.x+100, row.y, row.z, "THREAT AREA", fontsize=9)
            else:
                ax.arrow(row.x, row.y, 100, 80, head_width=20, head_length=20, fc=color, ec=color, alpha=0.5)
                ax.text(row.x+100, row.y+80, "THREAT AREA", fontsize=9)

    # final touches
    if not three_d:
        ax.legend(loc='upper left')
    else:
        ax.set_xlim(-600,600); ax.set_ylim(-600,600); ax.set_zlim(0,500)
        ax.legend(loc='upper left')

    return fig, frame, drone_counts, events

#######################################
# 4. Streamlit App Layout & Logic
#######################################
st.set_page_config(page_title="Drone Intelligence Dashboard", layout="wide")
st.title("Drone Intelligence Dashboard")
st.markdown("""
Monitor enemy drone activity relative to frontline units.  
Use the sidebar to replay or step through the simulation.
""")

# Controls
st.sidebar.header("Command Center Controls")
t_slider = st.sidebar.slider("Time (s)", 0, int(df.time.max()), 0, 1)
play      = st.sidebar.button("▶️ Play")
speed     = st.sidebar.number_input("Speed (s/frame)", 0.1, 5.0, 1.0, 0.1)

col_radar, col_summary = st.columns([2,1])
radar2d_pl, radar3d_pl = col_radar.empty(), col_radar.empty()
summary_md, log_md   = col_summary.empty(), col_summary.empty()

def render(t):
    fig2, frm, counts, evs = plot_frame(t, three_d=False)
    radar2d_pl.pyplot(fig2)
    fig3, *_ = plot_frame(t, three_d=True)
    radar3d_pl.pyplot(fig3)
    # summary
    lines = ["**Current Drone Counts:**"] + [f"- {k}: {v}" for k,v in counts.items()]
    lines += ["\n**Nearest per Unit:**"]
    for name,pos in infantry_positions.items():
        nearest, dmin = None, float('inf')
        for _, r in frm.iterrows():
            d = distance((r.x,r.y,r.z), pos)
            if d<dmin: nearest, dmin = f"{r['type']} ({r.id})", d
        lines.append(f"- {name}: {nearest or '—'} at {dmin:.1f}m")
    summary_md.markdown("\n".join(lines))
    log_md.markdown("\n".join(f"- {e}" for e in evs) or "No alerts.")

if play:
    for t in range(t_slider, int(df.time.max())+1):
        render(t)
        time.sleep(speed)
else:
    render(t_slider)
