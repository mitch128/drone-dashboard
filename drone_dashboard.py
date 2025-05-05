import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time, math

@st.cache_data
def make_dummy_data():
    np.random.seed(0)
    times = np.arange(0, 11)
    rows = []
    for t in times:
        # Shahed
        rows.append(dict(time=t, id='D1', type='Shahed',
                         x=-400 + 50*t + np.random.normal(0,5),
                         y=-300 + 40*t + np.random.normal(0,5),
                         z=100 + 10*np.sin(0.5*t)))
        # DJI Mavic
        ang = 0.4 * t
        rows.append(dict(time=t, id='D2', type='DJI Mavic',
                         x=100 + 12*np.sin(ang) + np.random.normal(0,2),
                         y=150 + 12*np.cos(ang) + np.random.normal(0,2),
                         z=50  + 5*np.sin(0.3*t)))
        # Recon
        rows.append(dict(time=t, id='D3', type='Recon',
                         x=-200 + 6*t + 4*np.sin(0.7*t) + np.random.normal(0,3),
                         y=300  - 3*t + 4*np.cos(0.7*t) + np.random.normal(0,3),
                         z=200 + np.random.normal(0,1)))
    return pd.DataFrame(rows)

df = make_dummy_data()
infantry = {"Alpha HQ": (0,0,0),}
COLORS = {'Shahed':'red','DJI Mavic':'blue','Recon':'green'}

def distance(a,b): return math.dist(a,b)

def plot_frame(t, three_d=False):
    # set up figure
    if three_d:
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(7,7))
        ax.set_facecolor('#EAEAEA')
        ax.set_xlim(-600,600); ax.set_ylim(-600,600)
        for r in (100,250,500):
            c = plt.Circle((0,0),r,fill=False,ls='--',color='gray',lw=1)
            ax.add_patch(c); ax.text(r-15,0,f"{r}m",fontsize=8,color='gray')

    ax.set_title(f"Drone Tracker t={t}s", fontsize=14)
    # infantry
    for name,(ix,iy,iz) in infantry.items():
        if three_d:
            ax.scatter(ix,iy,iz,c='k',marker='s',s=60)
            ax.text(ix,iy,iz,name,fontsize=8)
        else:
            ax.plot(ix,iy,'ks',ms=8); ax.text(ix+8,iy+8,name,fontsize=8)

    # 1) plot every drone's trajectory up to t
    for drone_id in df.id.unique():
        path = df[(df.id==drone_id) & (df.time<=t)].sort_values('time')
        if len(path)>1:
            dtype = path.type.iloc[0]
            c = COLORS[dtype]
            # fade: older segments more transparent
            segment_alphas = np.linspace(0.2, 0.6, len(path)-1)
            for i in range(len(path)-1):
                x0,y0,z0 = path.iloc[i][['x','y','z']]
                x1,y1,z1 = path.iloc[i+1][['x','y','z']]
                if three_d:
                    ax.plot([x0,x1],[y0,y1],[z0,z1], ls=':', color=c, alpha=segment_alphas[i])
                else:
                    ax.plot([x0,x1],[y0,y1],      ls=':', color=c, alpha=segment_alphas[i])

    # 2) overlay current positions + events
    now = df[df.time==t]
    counts, events, seen = {}, [], set()
    for _, r in now.iterrows():
        counts[r.type] = counts.get(r.type,0)+1
        c = COLORS[r.type]
        # scatter + label
        if three_d:
            ax.scatter(r.x, r.y, r.z, c=c, s=80,
                       label=r.type if r.type not in seen else "")
            ax.text(r.x, r.y, r.z, f"{r.id}", fontsize=7)
        else:
            ax.plot(r.x, r.y, 'o', c=c, ms=8,
                    label=r.type if r.type not in seen else "")
            ax.text(r.x+5, r.y+5, f"{r.id}", fontsize=7)
        seen.add(r.type)
        # threat arrow
        if r.type=='Shahed':
            events.append(f"ALERT {r.id} at ({r.x:.0f},{r.y:.0f})")
            if three_d:
                ax.quiver(r.x,r.y,r.z,100,80,0,color=c,alpha=0.5)
            else:
                ax.arrow(r.x,r.y,100,80,head_width=10,head_length=10,fc=c,ec=c,alpha=0.5)

    if not three_d:
        ax.legend(loc='upper left', fontsize=8)
    else:
        ax.set_xlim(-600,600); ax.set_ylim(-600,600); ax.set_zlim(0,500)
        ax.legend(loc='upper left', fontsize=8)

    return fig, now, counts, events

# --- Streamlit layout ---
st.title("Drone Intelligence Dashboard")
st.sidebar.header("Controls")
t0 = int(df.time.max())
t = st.sidebar.slider("Time", 0, t0, 0, 1)
play = st.sidebar.button("Play ▶️")
spd  = st.sidebar.number_input("Speed (s/frame)",0.1,5.0,1.0,0.1)

col2, col3 = st.columns(2)
ph2, ph3 = col2.empty(), col3.empty()
sum_p = st.sidebar.empty(); log_p = st.sidebar.empty()

def render(tt):
    f2, fr, cnt, ev = plot_frame(tt, three_d=False)
    ph2.pyplot(f2)
    f3, *_ = plot_frame(tt, three_d=True)
    ph3.pyplot(f3)
    # summary
    lines = ["**Counts:**"] + [f"- {k}: {v}" for k,v in cnt.items()] + ["\n**Closest:**"]
    for name,pos in infantry.items():
        dmin,drone = float('inf'),'—'
        for _,x in fr.iterrows():
            d = distance((x.x,x.y,x.z), pos)
            if d<dmin: dmin,drone = d,f"{x.id}"
        lines.append(f"- {name}: {drone} @ {dmin:.1f}m")
    sum_p.markdown("\n".join(lines))
    log_p.markdown("\n".join(f"- {e}" for e in ev) or "No alerts.")

if play:
    for tt in range(t, t0+1):
        render(tt); time.sleep(spd)
else:
    render(t)
