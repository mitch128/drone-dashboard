import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D projection
import time
import math

#######################################
# 1. Generate More Realistic Dummy Drone Data in 3D
#######################################
def make_dummy_data():
    np.random.seed(42)
    timestamps = np.arange(0, 31)
    rows = []
    # Drone D1: High-threat (Shahed)
    start_D1 = np.array([-600, -500, 200])
    base_velocity_D1 = np.array([30, 35, -4])
    for t in timestamps:
        deviation = np.array([15 * np.sin(0.1*t), 15 * np.cos(0.1*t), 5 * np.sin(0.05*t)])
        noise = np.random.normal(0, 2, 3)
        pos = start_D1 + base_velocity_D1 * t + deviation + noise
        rows.append({'time': t, 'id': 'D1', 'type': 'Shahed', 'x': pos[0], 'y': pos[1], 'z': pos[2]})
    # Drone D2: Recon (DJI Mavic)
    start_D2 = np.array([200, 400, 100])
    for t in timestamps:
        angle = 0.2 * t
        radius = 50 + 0.5 * t
        pos = start_D2 + np.array([radius * np.cos(angle), radius * np.sin(angle), 2*t])
        noise = np.random.normal(0, 1.5, 3)
        pos += noise
        rows.append({'time': t, 'id': 'D2', 'type': 'DJI Mavic', 'x': pos[0], 'y': pos[1], 'z': pos[2]})
    # Drone D3: Surveillance (Recon)
    start_D3 = np.array([-300, 600, 150])
    for t in timestamps:
        base = start_D3 + np.array([10*t, -12*t, t])
        oscillation = np.array([8 * np.sin(0.15*t), 8 * np.cos(0.15*t), 3 * np.sin(0.2*t)])
        noise = np.random.normal(0, 1, 3)
        pos = base + oscillation + noise
        rows.append({'time': t, 'id': 'D3', 'type': 'Recon', 'x': pos[0], 'y': pos[1], 'z': pos[2]})
    df = pd.DataFrame(rows)
    for drone_id in df['id'].unique():
        mask = df['id'] == drone_id
        df.loc[mask, ['velocity_x','velocity_y','velocity_z']] = df[mask][['x','y','z']].diff().fillna(0)
    return df

df = make_dummy_data()

#######################################
# 2. Define Single Friendly Position in 3D (Central Unit)
#######################################
# Only one position in the middle
infantry_positions = {
    "Central Unit": (0, 0, 0)
}

#######################################
# Utility functions
#######################################
def distance_3d(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def compute_impact_zone(x, y, z, vx, vy, vz, seconds=5):
    proj_x = x + vx * seconds
    proj_y = y + vy * seconds
    proj_z = z + vz * seconds
    disp = math.sqrt((vx*seconds)**2 + (vy*seconds)**2 + (vz*seconds)**2)
    radius = 0.15 * disp + 15
    return proj_x, proj_y, proj_z, radius

#######################################
# 3D Plotting
#######################################
def plot_radar_frame_3d(t):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#F7F7F7')
    ax.set_title(f"3D Drone Tracker – Time = {t}s", fontsize=16)
    ax.set_xlim(-700,700); ax.set_ylim(-700,700); ax.set_zlim(0,400)
    xx, yy = np.meshgrid(np.linspace(-700,700,2), np.linspace(-700,700,2))
    ax.plot_surface(xx, yy, np.zeros_like(xx), color='lightgray', alpha=0.3)
    for name, (ix,iy,iz) in infantry_positions.items():
        ax.scatter(ix,iy,iz,c='k',marker='^',s=100)
        ax.text(ix+10,iy+10,iz+10,name,fontsize=10,weight='bold')
    current = df[df.time==t]
    for drone_id in df['id'].unique():
        hist = df[(df.id==drone_id)&(df.time<=t)]
        ax.plot(hist.x, hist.y, hist.z, linestyle='dotted', color='gray', alpha=0.7)
    info, events = {}, []
    for _,row in current.iterrows():
        c = 'red' if row.type=='Shahed' else 'blue' if row.type=='DJI Mavic' else 'green'
        ax.scatter(row.x,row.y,row.z,c=c,marker='o',s=80,alpha=0.9)
        vmag = math.sqrt(row.velocity_x**2+row.velocity_y**2+row.velocity_z**2)
        ax.text(row.x+10,row.y+10,row.z+10,f"{row.id}\n{row.type}\nV:{vmag:.1f}",fontsize=9,color=c)
        if row.type=='Shahed':
            px,py,pz,rad = compute_impact_zone(row.x,row.y,row.z,row.velocity_x,row.velocity_y,row.velocity_z)
            ax.quiver(row.x,row.y,row.z,px-row.x,py-row.y,pz-row.z,color='red',arrow_length_ratio=0.15,alpha=0.7)
            u = np.linspace(0,2*np.pi,20); v = np.linspace(0,np.pi,10)
            xs = px + rad*np.outer(np.cos(u), np.sin(v))
            ys = py + rad*np.outer(np.sin(u), np.sin(v))
            zs = pz + rad*np.outer(np.ones_like(u), np.cos(v))
            ax.plot_wireframe(xs,ys,zs,color='red',alpha=0.3)
            ax.text(px+10,py+10,pz+10,"Projected Impact Zone",fontsize=8,color='red')
            events.append(f"ALERT: {row.id} impact at ({px:.1f},{py:.1f},{pz:.1f}) r={rad:.1f}m")
    return fig, current, info, events

#######################################
# 2D Plotting
#######################################
def plot_radar_frame_2d(t):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_facecolor('#F7F7F7')
    ax.set_title(f"2D Birds-eye Drone Tracker – Time = {t}s", fontsize=16)
    ax.set_xlim(-700,700); ax.set_ylim(-700,700)
    current = df[df.time==t]
    for name,(ix,iy,_) in infantry_positions.items():
        ax.plot(ix,iy,'ks',markersize=10)
        ax.text(ix+10,iy+10,name,fontsize=10,weight='bold')
        for r in (100,250,500):
            ax.add_patch(plt.Circle((ix,iy),r,fill=False,linestyle='dotted',alpha=0.5))
    events=[]; info={}
    for drone_id in df['id'].unique():
        hist = df[(df.id==drone_id)&(df.time<=t)]
        ax.plot(hist.x,hist.y,linestyle='dotted',color='gray',alpha=0.7)
    for _,row in current.iterrows():
        c='red' if row.type=='Shahed' else 'blue' if row.type=='DJI Mavic' else 'green'
        ax.plot(row.x,row.y,'o',color=c,markersize=8,alpha=0.9)
        vmag = math.sqrt(row.velocity_x**2+row.velocity_y**2+row.velocity_z**2)
        ax.text(row.x+10,row.y+10,f"{row.id}\nV:{vmag:.1f}",fontsize=8,color=c)
        if row.type=='Shahed':
            px,py,_,rad = compute_impact_zone(row.x,row.y,row.z,row.velocity_x,row.velocity_y,row.velocity_z)
            ax.arrow(row.x,row.y,px-row.x,py-row.y,head_width=15,head_length=15,fc='red',ec='red',alpha=0.7)
            ax.add_patch(plt.Circle((px,py),rad,fill=True,alpha=0.2))
            ax.text(px+10,py+10,"Impact Zone",fontsize=8,color='red')
            events.append(f"ALERT: {row.id} 2D impact at ({px:.1f},{py:.1f}) r={rad:.1f}m")
    ax.grid(True,linestyle='--',alpha=0.5)
    return fig, current, info, events

#######################################
# Streamlit UI – 2D Left, 3D Right
#######################################
st.set_page_config(page_title="Drone Intelligence Dashboard", layout="wide")
st.title("Drone Intelligence Dashboard (Multi-View)")
st.markdown("This tool monitors enemy drone activity in both 2D and 3D views. Use controls to play or step through the simulation.")
# Controls
st.sidebar.header("Controls")
t_slider = st.sidebar.slider("Time (s)",0,int(df.time.max()),0,1)
play = st.sidebar.button("Play")
speed = st.sidebar.number_input("Speed (s/frame)",0.1,5.0,0.75,0.1)
# Layout
col2d, col3d = st.columns(2)
summary_container = st.container()
with col2d:
    st.subheader("2D View")
    placeholder2d = st.empty()
with col3d:
    st.subheader("3D View")
    placeholder3d = st.empty()
with summary_container:
    st.subheader("Summary")
    summary_box = st.empty()
    st.subheader("Events")
    events_box = st.empty()
# Simulation loop (unchanged)
if play:
    for t in range(t_slider, int(df.time.max())+1):
        f3,cf3,_,e3 = plot_radar_frame_3d(t)
        f2,cf2,_,e2 = plot_radar_frame_2d(t)
        placeholder2d.pyplot(f2)
        placeholder3d.pyplot(f3)
        summary_box.markdown(f"**Time:** {t}s")
        events = e3+e2
        events_box.markdown("\n".join(f"- {e}" for e in events) if events else "No events.")
        time.sleep(speed)
else:
    f3,cf3,_,e3 = plot_radar_frame_3d(t_slider)
    f2,cf2,_,e2 = plot_radar_frame_2d(t_slider)
    placeholder2d.pyplot(f2)
    placeholder3d.pyplot(f3)
    summary_box.markdown(f"**Time:** {t_slider}s")
    events = e3+e2
    events_box.markdown("\n".join(f"- {e}" for e in events) if events else "No events.")
