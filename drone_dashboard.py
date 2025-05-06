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
    start = np.array([-150, -100, 50]); v0 = np.array([3, 4, -0.5])
    for t in times:
        dev = np.array([5*np.sin(0.1*t), 5*np.cos(0.1*t), 2*np.sin(0.05*t)])
        pos = start + v0*t + dev + np.random.normal(0,0.5,3)
        rows.append({'time':t,'id':'D1','x':pos[0],'y':pos[1],'z':pos[2]})
    # D2
    start = np.array([50,80,50])
    for t in times:
        ang = 0.15*t + 0.05*np.sin(0.1*t)
        rad = 20 + 0.3*t + 2*np.sin(0.2*t)
        pos = start + np.array([rad*np.cos(ang), rad*np.sin(ang), 1.5*t]) + np.random.normal(0,0.3,3)
        rows.append({'time':t,'id':'D2','x':pos[0],'y':pos[1],'z':pos[2]})
    # D3
    start = np.array([-80,120,60])
    for t in times:
        base = start + np.array([2*t,-2.5*t,0.5*t])
        osc = np.array([3*np.sin(0.15*t),3*np.cos(0.15*t),1.5*np.sin(0.2*t)])
        pos = base + osc + np.random.normal(0,0.2,3)
        rows.append({'time':t,'id':'D3','x':pos[0],'y':pos[1],'z':pos[2]})
    df = pd.DataFrame(rows)
    for d in df['id'].unique():
        m = df['id']==d
        df.loc[m, ['vx','vy','vz']] = df[m][['x','y','z']].diff().fillna(0)
    return df

# Load data
df = make_dummy_data()

# Friendly assets (squares)
friendly_assets = [(80, -50), (-100, 100)]  # fixed coords
# Single ring at 100m
ring = 100
# Marker and shading sizes per drone (meters)
marker_size = {'D1':5, 'D2':4, 'D3':3}
shading_radius = {'D1': marker_size['D1'] * 2.0, 'D2': marker_size['D2'] * 1.8, 'D3': marker_size['D3'] * 1.5}

# Impact calculation
def compute_impact(x,y,z,vx,vy,vz,sec=5):
    px,py,pz = x+vx*sec, y+vy*sec, z+vz*sec
    dist = math.sqrt((vx*sec)**2+(vy*sec)**2+(vz*sec)**2)
    return px,py,pz, 0.15*dist+5

#######################################
# Plot Functions with center marker
#######################################

def plot_3d(t):
    fig = plt.figure(figsize=(6,6)); ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-200,200); ax.set_ylim(-200,200); ax.set_zlim(0,250)
    ax.set_title(f"Time = {t}s")
    # 100m ring on ground
    theta = np.linspace(0,2*np.pi,200)
    x_ring = ring * np.cos(theta)
    y_ring = ring * np.sin(theta)
    ax.plot(x_ring, y_ring, np.zeros_like(theta), '--', color='black')
    ax.text(ring*np.cos(np.pi/4), ring*np.sin(np.pi/4), 0, f"{ring}m", color='black')
    # center/base marker
    ax.scatter(0, 0, 0, marker='*', c='black', s=100)
    ax.text(0, 0, 0, " Base", color='black')
    # assets
    for axx,ayy in friendly_assets:
        ax.scatter(axx, ayy, 0, marker='s', c='black', s=80)
    # history
    for d in df['id'].unique(): hist = df[(df.id==d)&(df.time<=t)]; ax.plot(hist.x, hist.y, hist.z, ':', color='gray')
    # current positions with shading
    for _,r in df[df.time==t].iterrows():
        c = 'red' if r.id=='D1' else 'blue' if r.id=='D2' else 'green'
        # shaded sphere
        u,v = np.mgrid[0:2*np.pi:12j,0:np.pi:6j]
        rad = shading_radius[r.id]
        xs = r.x + rad*np.cos(u)*np.sin(v)
        ys = r.y + rad*np.sin(u)*np.sin(v)
        zs = r.z + rad*np.cos(v)
        ax.plot_surface(xs, ys, zs, color=c, alpha=0.2)
        # smaller marker
        ms = marker_size[r.id]
        ax.scatter(r.x, r.y, r.z, c=c, s=ms*4)  # scaled for matplotlib
        ax.text(r.x+5, r.y+5, r.z+5, r.id, color=c)
    ax.set_box_aspect([1,1,0.8])
    return fig


def plot_2d(t):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-200,200); ax.set_ylim(-200,200)
    ax.set_title(f"Time = {t}s")
    # 100m ring and label
    circ = plt.Circle((0,0), ring, fill=False, linestyle='--', color='black')
    ax.add_patch(circ)
    ax.text(ring/np.sqrt(2), ring/np.sqrt(2), f"{ring}m", color='black')
    # center/base marker
    ax.scatter(0, 0, marker='*', c='black', s=100)
    ax.text(0, 0, " Base", color='black')
    # assets
    for axx,ayy in friendly_assets:
        ax.scatter(axx, ayy, marker='s', c='black', s=80)
    # history
    for d in df['id'].unique(): hist = df[(df.id==d)&(df.time<=t)]; ax.plot(hist.x, hist.y, ':', color='gray')
    # current with shading
    for _,r in df[df.time==t].iterrows():
        c = 'red' if r.id=='D1' else 'blue' if r.id=='D2' else 'green'
        rad = shading_radius[r.id]
        ax.add_patch(plt.Circle((r.x, r.y), rad, color=c, alpha=0.2))
        ms = marker_size[r.id]
        ax.plot(r.x, r.y, 'o', color=c, markersize=ms*2)
        ax.text(r.x+5, r.y+5, r.id, color=c)
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax.set_aspect('equal')
    return fig

#######################################
# UI Layout
#######################################

st.title("Drone SkyView - v3.22")
# Controls in thin column
top_cols = st.columns([1,10,10])
with top_cols[0]:
    t = st.slider("Time", 0, int(df.time.max()), 0)
    play = st.button("▶️")

# Placeholders
placeholder2d = top_cols[1].empty()
placeholder3d = top_cols[2].empty()
summary_box = st.empty()

# Render frame
def render(t0):
    fig2 = plot_2d(t0)
    fig3 = plot_3d(t0)
    placeholder2d.pyplot(fig2)
    placeholder3d.pyplot(fig3)
    counts = df[df.time==t0]['id'].value_counts().to_dict()
    summary_box.markdown("**Counts:** " + ", ".join(f"{k}:{v}" for k,v in counts.items()))

# Main loop
if play:
    for t0 in range(t, int(df.time.max())+1):
        render(t0)
        time.sleep(0.5)
else:
    render(t)
