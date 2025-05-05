import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl\_toolkits.mplot3d import Axes3D
import time, math

# --- Dummy Data Generation ---

@st.cache\_data
def make\_dummy\_data():
np.random.seed(42)
times = np.arange(0, 61)  # simulate 1 minute at 1s intervals
rows = \[]
\# Define waypoints and behavior per drone
\# D1: Shahed missile: straight approach towards Bravo FOB
start1 = np.array(\[-500, -400, 150])
target1 = np.array(\[200, 100, 0])  # Bravo FOB
v1 = (target1 - start1) / times\[-1]
\# D2: DJI Mavic: random walk with hover segments
pos2 = np.array(\[100.0, 150.0, 60.0])  # FIX: use floats
hover\_times = set(np.random.choice(times, size=10, replace=False))
\# D3: Recon UAV: patrol circle centered on Charlie OP
center3 = np.array(\[-150, -100, 0])
radius3 = 100
ang\_rate3 = 2 \* np.pi / times\[-1]

```
for t in times:
    # Shahed
    pos1 = start1 + v1 * t + np.random.normal(0, 3, 3)
    rows.append(dict(time=t, id='D1', type='Shahed',
                     x=pos1[0], y=pos1[1], z=max(pos1[2], 5)))
    # DJI Mavic
    if t in hover_times:
        pos2 += np.random.normal(0, 0.5, 3)
    else:
        theta = np.random.uniform(0, 2*np.pi)
        step = np.array([np.cos(theta), np.sin(theta), np.random.uniform(-0.2, 0.2)]) * 2
        pos2 += step
    pos2[2] = np.clip(pos2[2], 20, 120)
    rows.append(dict(time=t, id='D2', type='DJI Mavic', x=pos2[0], y=pos2[1], z=pos2[2]))
    # Recon
    theta3 = ang_rate3 * t
    x3 = center3[0] + radius3 * np.cos(theta3) + np.random.normal(0, 2)
    y3 = center3[1] + radius3 * np.sin(theta3) + np.random.normal(0, 2)
    z3 = 200 + np.random.normal(0, 5)
    rows.append(dict(time=t, id='D3', type='Recon', x=x3, y=y3, z=z3))

return pd.DataFrame(rows)
```

# Load data

df = make\_dummy\_data()
infantry = {"Alpha HQ": (0, 0, 0), "Bravo FOB": (200, 100, 0), "Charlie OP": (-150, -100, 0)}
COLORS = {'Shahed': 'red', 'DJI Mavic': 'blue', 'Recon': 'green'}

# Utility: distance

def distance(a, b):
return math.dist(a, b)

# Compute projected impact if applicable

def compute\_impact(r, last):
v = np.array(\[r.x - last.x, r.y - last.y, r.z - last.z])
if r.type == 'DJI Mavic' or np.linalg.norm(v\[:2]) < 1e-1:
return None
if abs(v\[2]) < 1e-2:
t\_proj = r.z / 5.0
else:
t\_proj = r.z / -v\[2]
impact = np.array(\[r.x + v\[0] \* t\_proj, r.y + v\[1] \* t\_proj, 0])
return impact

# Plotting

def plot\_frame(t, three\_d=False):
if three\_d:
fig = plt.figure(figsize=(7, 7))
ax = fig.add\_subplot(111, projection='3d')
else:
fig, ax = plt.subplots(figsize=(7, 7))
ax.set\_facecolor('#EAEAEA')
ax.set\_xlim(-600, 600)
ax.set\_ylim(-600, 600)
for r in (100, 250, 500):
c = plt.Circle((0, 0), r, fill=False, ls='--', color='gray', lw=1)
ax.add\_patch(c)
ax.text(r - 15, 0, f"{r}m", fontsize=8, color='gray')

```
ax.set_title(f"Drone Tracker t={t}s", fontsize=14)

for name, (ix, iy, iz) in infantry.items():
    if three_d:
        ax.scatter(ix, iy, iz, c='k', marker='s', s=60)
        ax.text(ix, iy, iz, name, fontsize=8)
    else:
        ax.plot(ix, iy, 'ks', ms=8)
        ax.text(ix + 8, iy + 8, name, fontsize=8)

for drone_id in df.id.unique():
    path = df[(df.id == drone_id) & (df.time <= t)].sort_values('time')
    if len(path) > 1:
        c = COLORS[path.type.iloc[0]]
        if three_d:
            ax.plot(path.x, path.y, path.z, ls=':', color=c, alpha=0.5)
        else:
            ax.plot(path.x, path.y, ls=':', color=c, alpha=0.5)

now = df[df.time == t]
events = []
seen = set()
for _, r in now.iterrows():
    c = COLORS[r.type]
    if three_d:
        ax.scatter(r.x, r.y, r.z, c=c, s=80, label=r.type if r.type not in seen else "")
        ax.text(r.x, r.y, r.z, f"{r.id}", fontsize=7)
    else:
        ax.plot(r.x, r.y, 'o', c=c, ms=8, label=r.type if r.type not in seen else "")
        ax.text(r.x + 5, r.y + 5, f"{r.id}", fontsize=7)
    seen.add(r.type)
    last = df[(df.id == r.id) & (df.time == t - 1)].iloc[0] if t > 0 else r
    impact = compute_impact(r, last)
    if impact is not None:
        if three_d:
            theta = np.linspace(0, 2 * np.pi, 50)
            x_c = impact[0] + 20 * np.cos(theta)
            y_c = impact[1] + 20 * np.sin(theta)
            z_c = np.zeros_like(theta)
            ax.plot(x_c, y_c, z_c, ls='--', alpha=0.7)
        else:
            circle = plt.Circle((impact[0], impact[1]), 20, fill=False, ls='--', alpha=0.7)
            ax.add_patch(circle)
    if r.type == 'Shahed':
        events.append(f"ALERT {r.id} at ({r.x:.0f},{r.y:.0f})")
        if three_d:
            ax.quiver(r.x, r.y, r.z, 50, 40, 0, color=c, alpha=0.5)
        else:
            ax.arrow(r.x, r.y, 50, 40, head_width=5, head_length=5, fc=c, ec=c, alpha=0.5)

if not three_d:
    ax.legend(loc='upper left', fontsize=8)
else:
    ax.set_xlim(-600, 600)
    ax.set_ylim(-600, 600)
    ax.set_zlim(0, 500)
    ax.legend(loc='upper left', fontsize=8)

return fig, now, events
```

# Streamlit UI

st.set\_page\_config(page\_title="Drone Dashboard", layout="wide")
st.title("Drone Intelligence Dashboard")
st.sidebar.header("Controls")
t\_max = int(df.time.max())
t = st.sidebar.slider("Time", 0, t\_max, 0, 1)
play = st.sidebar.button("Play ▶️")
spd = st.sidebar.number\_input("Speed (s/frame)", 0.1, 5.0, 1.0, 0.1)
col2, col3 = st.columns(2)
ph2, ph3 = col2.empty(), col3.empty()
sum\_p = st.sidebar.empty()
log\_p = st.sidebar.empty()

def render(tt):
f2, now2, events = plot\_frame(tt, three\_d=False)
ph2.pyplot(f2)
f3, \_, \_ = plot\_frame(tt, three\_d=True)
ph3.pyplot(f3)

```
counts = now2.type.value_counts().to_dict()
lines = ["**Counts:**"] + [f"- {k}: {v}" for k, v in counts.items()] + ["\n**Events:**"] + ([f"- {e}" for e in events] or ["- No alerts."])
sum_p.markdown("\n".join(lines))

dlines = ["\n**Closest:**"]
for name, pos in infantry.items():
    df_now = now2.copy()
    df_now['dist'] = df_now.apply(lambda r: distance((r.x, r.y, r.z), pos), axis=1)
    row = df_now.loc[df_now.dist.idxmin()]
    dlines.append(f"- {name}: {row.id} @ {row.dist:.1f}m")
log_p.markdown("\n".join(dlines))
```

if play:
for tt in range(t, t\_max + 1):
render(tt)
time.sleep(spd)
else:
render(t)
