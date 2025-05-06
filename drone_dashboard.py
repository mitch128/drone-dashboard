import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import time
import math

# --- Page Config ---
st.set_page_config(page_title="Drone SkyView - v3.22", layout="wide")

#######################################
# Data Generation
#######################################
def make_dummy_data():
    np.random.seed(42)
    times = np.arange(0, 31)
    rows = []
    # D1
    start = np.array([-600, -500, 200]); v0 = np.array([30,35,-4])
    for t in times:
        dev = np.array([15*np.sin(0.1*t), 15*np.cos(0.1*t), 5*np.sin(0.05*t)])
        pos = start + v0*t + dev + np.random.normal(0,2,3)
        rows.append({'time':t,'id':'D1','x':pos[0],'y':pos[1],'z':pos[2]})
    # D2
    start = np.array([200,400,100])
    for t in times:
        ang = 0.15*t + 0.05*np.sin(0.1*t)
        rad = 50 + 0.5*t + 5*np.sin(0.2*t)
        pos = start + np.array([rad*np.cos(ang), rad*np.sin(ang), 2*t]) + np.random.normal(0,1.5,3)
        rows.append({'time':t,'id':'D2','x':pos[0],'y':pos[1],'z':pos[2]})
    # D3
    start = np.array([-300,600,150])
    for t in times:
        base = start + np.array([10*t,-12*t,t])
        osc = np.array([8*np.sin(0.15*t),8*np.cos(0.15*t),3*np.sin(0.2*t)])
        pos = base + osc + np.random.normal(0,1,3)
        rows.append({'time':t,'id':'D3','x':pos[0],'y':pos[1],'z':pos[2]})
    df = pd.DataFrame(rows)
    for d in df['id'].unique():
        m = df['id']==d
        df.loc[m, ['vx','vy','vz']] = df[m][['x','y','z']].diff().fillna(0)
    return df

df = make_dummy_data()

# Friendly assets (squares)
friendly_assets = [(100, -100), (-150, 200)]  # fixed coords
# Concentric radii
rings = [100, 250]
# Sphere sizes per drone
droner = {'D1':50,'D2':40,'D3':30}

# Impact calculation
def compute_impact(x,y,z,vx,vy,vz,sec=5):
    px,py,pz = x+vx*sec, y+vy*sec, z+vz*sec
    dist = math.sqrt((vx*sec)**2+(vy*sec)**2+(vz*sec)**2)
    return px,py,pz, 0.15*dist+15

#######################################
# Plot Functions
#######################################

def plot_3d(t):
    fig = plt.figure(figsize=(6,6)); ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-700,700); ax.set_ylim(-700,700); ax.set_zlim(0,400)
    # rings on ground
    theta = np.linspace(0,2*np.pi,200)
    for r in rings:
        ax.plot(r*np.cos(theta), r*np.sin(theta), np.zeros_like(theta), '--', color='black')
        # label
        ax.text(r*np.cos(np.pi/4), r*np.sin(np.pi/4), 0, f"{r}m", color='black')
    # assets
    for axx,ayy in friendly_assets:
        ax.scatter(axx, ayy, 0, marker='s', c='black', s=80)
    # historical paths
    for d in df['id'].unique(): hist = df[(df.id==d)&(df.time<=t)]; ax.plot(hist.x, hist.y, hist.z, ':', color='gray')
    events=[]
    # current drones
    for _,r in df[df.time==t].iterrows():
        c = 'red' if r.id=='D1' else 'blue' if r.id=='D2' else 'green'
        # sphere
        u,v = np.mgrid[0:2*np.pi:12j,0:np.pi:6j]
        xs = r.x + droner[r.id]*np.cos(u)*np.sin(v)
        ys = r.y + droner[r.id]*np.sin(u)*np.sin(v)
        zs = r.z + droner[r.id]*np.cos(v)
        ax.plot_surface(xs, ys, zs, color=c, alpha=0.2)
        ax.scatter(r.x, r.y, r.z, c=c, s=20)
        ax.text(r.x+10, r.y+10, r.z+10, r.id, color=c)
    ax.set_box_aspect([1,1,0.5])
    return fig


def plot_2d(t):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-700,700); ax.set_ylim(-700,700)
    # rings and labels
    for r in rings:
        circ = plt.Circle((0,0), r, fill=False, linestyle='--', color='black')
        ax.add_patch(circ)
        ax.text(r/np.sqrt(2), r/np.sqrt(2), f"{r}m")
    # assets
    for axx,ayy in friendly_assets:
        ax.scatter(axx, ayy, marker='s', c='black', s=80)
    # historical
    for d in df['id'].unique(): hist = df[(df.id==d)&(df.time<=t)]; ax.plot(hist.x, hist.y, ':', color='gray')
    # current
    for _,r in df[df.time==t].iterrows():
        c = 'red' if r.id=='D1' else 'blue' if r.id=='D2' else 'green'
        ax.add_patch(plt.Circle((r.x, r.y), droner[r.id], color=c, alpha=0.2))
        ax.plot(r.x, r.y, 'o', color=c, markersize=4)
        ax.text(r.x+10, r.y+10, r.id, color=c)
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax.set_aspect('equal')
    return fig

#######################################
# UI Layout
#######################################
st.title("Drone SkyView - v3.22")
# Controls in a narrow column
top_cols = st.columns([1,10,10])
with top_cols[0]:
    t = st.slider("Time", 0, int(df.time.max()), 0)
    play = st.button("▶️")

# Placeholders
placeholder2d = top_cols[1].empty()
placeholder3d = top_cols[2].empty()
summary_box = st.empty()

# Render function
def render(t0):
    fig2 = plot_2d(t0)
    fig3 = plot_3d(t0)
    placeholder2d.pyplot(fig2)
    placeholder3d.pyplot(fig3)
    # simple summary
    counts = df[df.time==t0]['id'].value_counts().to_dict()
    summary_box.markdown("**Drone Counts:** " + ", ".join(f"{k}:{v}" for k,v in counts.items()))

# Loop
if play:
    for t0 in range(t, int(df.time.max())+1):
        render(t0)
        time.sleep(0.5)
else:
    render(t)
