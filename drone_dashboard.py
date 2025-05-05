import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time, math

#######################################
# 1. Generate More Realistic Dummy Drone Data
#######################################
@st.cache_data
def make_dummy_data():
    np.random.seed(0)  # reproducible “noise”
    timestamps = np.arange(0, 11)  # 0 through 10 seconds
    rows = []
    for t in timestamps:
        # Shahed: mostly straight but with slight jitter + altitude oscillation
        x1 = -400 + 50 * t + np.random.normal(0, 8)
        y1 = -300 + 40 * t + np.random.normal(0, 8)
        z1 = 100 + 15 * np.sin(0.5 * t) + np.random.normal(0, 2)
        rows.append(dict(time=t, id='D1', type='Shahed',     x=x1, y=y1, z=z1))

        # DJI Mavic: small circle + jitter + gentle climb/descent
        angle = 0.4 * t
        x2 = 100 + 15 * np.sin(angle) + np.random.normal(0, 2)
        y2 = 150 + 15 * np.cos(angle) + np.random.normal(0, 2)
        z2 = 50 + 10 * np.sin(0.3 * t)
        rows.append(dict(time=t, id='D2', type='DJI Mavic',  x=x2, y=y2, z=z2))

        # Recon: diagonal sweep with occasional lateral wiggle
        x3 = -200 + 6 * t + 5 * np.sin(0.7 * t) + np.random.normal(0, 3)
        y3 = 300 - 3 * t + 5 * np.cos(0.7 * t) + np.random.normal(0, 3)
        z3 = 200 + np.random.normal(0, 1)
        rows.append(dict(time=t, id='D3', type='Recon',      x=x3, y=y3, z=z3))

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

TYPE_COLORS = {'Shahed':'red','DJI Mavic':'blue','Recon':'green'}

def distance(p1, p2):
    return math.dist(p1, p2)

#######################################
# 3. Plotting Helper (2D + 3D)
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
    # set up
    if three_d:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_facecolor('#EAEAEA')
        ax.set_xlim(-600,600); ax.set_ylim(-600,600)
        add_rings(ax)

    ax.set_title(f"Live Drone Tracker – t = {t}s", fontsize=16)
    add_infantry(ax, is3d=three_d)

    # 1) plot full dotted trajectories for *every* drone
    past = df[df.time <= t]
    for drone_id, path in past.groupby('id'):
        dtype = path['type'].iloc[0]
        color = TYPE_COLORS[dtype]
        xs, ys, zs = path.x.values, path.y.values, path.z.values
        if three_d:
            ax.plot(xs, ys, zs, linestyle=':', color=color, alpha=0.4, linewidth=1)
        else:
            ax.plot(xs, ys, linestyle=':', color=color, alpha=0.4, linewidth=1)

    # 2) overlay current positions, arrows, and build counts/events
    current = df[df.time == t]
    counts, events, seen = {}, [], set()
    for _, row in current.iterrows():
        counts[row.type] = counts.get(row.type,0) + 1
        c = TYPE_COLORS[row.type]

        # scatter + label
        if three_d:
            ax.scatter(row.x, row.y, row.z, c=c, s=80,
                       label=row.type if row.type not in seen else "")
            ax.text(row.x, row.y, row.z, f"{row.id}\n({row.type})", fontsize=8)
        else:
            ax.plot(row.x, row.y, 'o', c=c, markersize=10,
                    label=row.type if row.type not in seen else "")
            ax.text(row.x+8, row.y+8, f"{row.id}\n({row.type})", fontsize=8)
        seen.add(row.type)

        # threat arrow + log
        if row.type=='Shahed':
            events.append(f"ALERT: {row.id} at ({row.x:.0f},{row.y:.0f})")
            if three_d:
                ax.quiver(row.x, row.y, row.z, 100, 80, 0, color=c, alpha=0.5)
            else:
                ax.arrow(row.x, row.y, 100, 80, head_width=15, head_length=15,
                         fc=c, ec=c, alpha=0.5)

    # legend & axes limits
    if three_d:
        ax.set_xlim(-600,600); ax.set_ylim(-600,600); ax.set_zlim(0,500)
        ax.legend(loc='upper left')
    else:
        ax.legend(loc='upper left')

    return fig, current, counts, events

#######################################
# 4. Streamlit App
#######################################
st.set_page_config(page_title="Drone Dashboard", layout="wide")
st.title("Drone Intelligence Dashboard")
st.markdown("Monitor drone paths, counts, and threat alerts in 2D & 3D.")

# sidebar
st.sidebar.header("Controls")
t_slider = st.sidebar.slider("Time (s)", 0, int(df.time.max()), 0, 1)
play      = st.sidebar.button("▶️ Play")
speed     = st.sidebar.number_input("Speed (s/frame)", 0.1, 5.0, 1.0, 0.1)

# placeholders
col2d, col3d = st.columns([1,1])
place2d = col2d.empty()
place3d = col3d.empty()
summary_p = st.sidebar.empty()
log_p     = st.sidebar.empty()

def render(t):
    fig2, frame2, cnts, evs = plot_frame(t, three_d=False)
    place2d.pyplot(fig2)
    fig3, *_, = plot_frame(t, three_d=True)
    place3d.pyplot(fig3)

    # summary
    lines = ["**Drone Counts:**"] + [f"- {k}: {v}" for k,v in cnts.items()]
    lines += ["\n**Nearest to Units:**"]
    for name,pos in infantry_positions.items():
        closest, dmin = None, float('inf')
        for _, r in frame2.iterrows():
            d = distance((r.x,r.y,r.z), pos)
            if d<dmin:
                dmin, closest = d, f"{r.id} ({r.type})"
        lines.append(f"- {name}: {closest or '—'} at {dmin:.1f} m")
    summary_p.markdown("\n".join(lines))

    # events
    log_p.markdown("\n".join(f"- {e}" for e in evs) or "No alerts.")

if play:
    for t in range(t_slider, int(df.time.max())+1):
        render(t)
        time.sleep(speed)
else:
    render(t_slider)
