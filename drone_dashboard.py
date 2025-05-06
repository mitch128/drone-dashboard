import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import time
import math

# --- Streamlit Page Config ---
st.set_page_config(page_title="Drone Dashboard", layout="wide")

#######################################
# 1. Generate Dummy Drone Data
#######################################
def make_dummy_data():
    np.random.seed(42)
    timestamps = np.arange(0, 31)
    rows = []
    # D1
    start = np.array([-600, -500, 200]); v0 = np.array([30,35,-4])
    for t in timestamps:
        dev = np.array([15*np.sin(0.1*t), 15*np.cos(0.1*t), 5*np.sin(0.05*t)])
        pos = start + v0*t + dev + np.random.normal(0,2,3)
        rows.append({'time':t,'id':'D1','x':pos[0],'y':pos[1],'z':pos[2]})
    # D2
    start = np.array([200,400,100])
    for t in timestamps:
        ang = 0.15*t + 0.05*np.sin(0.1*t)
        rad = 50 + 0.5*t + 5*np.sin(0.2*t)
        pos = start + np.array([rad*np.cos(ang), rad*np.sin(ang), 2*t]) + np.random.normal(0,1.5,3)
        rows.append({'time':t,'id':'D2','x':pos[0],'y':pos[1],'z':pos[2]})
    # D3
    start = np.array([-300,600,150])
    for t in timestamps:
        base = start + np.array([10*t,-12*t,t])
        osc = np.array([8*np.sin(0.15*t),8*np.cos(0.15*t),3*np.sin(0.2*t)])
        pos = base + osc + np.random.normal(0,1,3)
        rows.append({'time':t,'id':'D3','x':pos[0],'y':pos[1],'z':pos[2]})
    df = pd.DataFrame(rows)
    for d in df['id'].unique():
        m = df['id']==d
        df.loc[m,['vx','vy','vz']] = df[m][['x','y','z']].diff().fillna(0)
    return df

df = make_dummy_data()

# Friendly unit
infantry_positions = {"Central Unit": (0,0,0)}
# Sphere sizes
sizes = {'D1':50,'D2':40,'D3':30}

# Impact zone util
def compute_impact(x,y,z,vx,vy,vz,sec=5):
    px,py,pz = x+vx*sec, y+vy*sec, z+vz*sec
    dist = math.sqrt((vx*sec)**2+(vy*sec)**2+(vz*sec)**2)
    r = 0.15*dist+15
    return px,py,pz,r

#######################################
# 3D Plot (white background)
#######################################
def plot_3d(t):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-700,700); ax.set_ylim(-700,700); ax.set_zlim(0,400)
    ax.set_title(f"Time = {t}s")
    # ground
    xx,yy = np.meshgrid(np.linspace(-700,700,2), np.linspace(-700,700,2))
    ax.plot_surface(xx,yy,np.zeros_like(xx),color='lightgray',alpha=0.3)
    # friendly
    for pos in infantry_positions.values(): ax.scatter(*pos, c='black', marker='^', s=150)
    # history
    for d in df['id'].unique(): hist = df[(df.id==d)&(df.time<=t)]; ax.plot(hist.x,hist.y,hist.z, linestyle='dotted', color='gray', alpha=0.5)
    events = []
    for _,r in df[df.time==t].iterrows():
        c = 'red' if r.id=='D1' else 'blue' if r.id=='D2' else 'green'
        # sphere
        u,v = np.mgrid[0:2*np.pi:12j,0:np.pi:6j]
        xs = r.x + sizes[r.id]*np.cos(u)*np.sin(v)
        ys = r.y + sizes[r.id]*np.sin(u)*np.sin(v)
        zs = r.z + sizes[r.id]*np.cos(v)
        ax.plot_surface(xs,ys,zs,color=c,alpha=0.2)
        # marker
        ax.scatter(r.x,r.y,r.z,c=c,s=20)
        ax.text(r.x+10,r.y+10,r.z+10, r.id, color=c)
        if r.id=='D1': px,py,pz,rad = compute_impact(r.x,r.y,r.z,r.vx,r.vy,r.vz); events.append(f"D1 impact at ({px:.0f},{py:.0f},{pz:.0f}) r={rad:.0f}m")
    return fig, events

#######################################
# 2D Plot (white background)
#######################################
def plot_2d(t):
    fig,ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(-700,700); ax.set_ylim(-700,700)
    ax.set_title(f"Time = {t}s")
    for pos in infantry_positions.values(): ax.scatter(pos[0],pos[1],c='black',marker='^',s=100)
    for d in df['id'].unique(): hist = df[(df.id==d)&(df.time<=t)]; ax.plot(hist.x,hist.y, linestyle='dotted', color='gray', alpha=0.5)
    events = []
    for _,r in df[df.time==t].iterrows():
        c = 'red' if r.id=='D1' else 'blue' if r.id=='D2' else 'green'
        ax.add_patch(plt.Circle((r.x,r.y), sizes[r.id], color=c, alpha=0.2))
        ax.plot(r.x,r.y,'o',color=c,markersize=2)
        ax.text(r.x+10,r.y+10, r.id, color=c)
        if r.id=='D1': px,py,_,rad = compute_impact(r.x,r.y,r.z,r.vx,r.vy,r.vz); events.append(f"D1 2D impact at ({px:.0f},{py:.0f}) r={rad:.0f}m")
    ax.grid(True, color='gray', linestyle='--', alpha=0.3)
    return fig, events

# --- Streamlit UI ---
st.title("Drone Intelligence Dashboard")
with st.sidebar:
    st.header("Controls")
    t = st.slider("Time (s)", 0, int(df.time.max()), 0)
    play = st.button("Play")
    delay = st.number_input("Speed s/frame", 0.1,5.0,0.75)

# placeholders
col1, col2 = st.columns(2)
placeholder2d = col1.empty()
placeholder3d = col2.empty()
events_placeholder = st.empty()

# render loop
def render_frame(time_val):
    fig2, ev2 = plot_2d(time_val)
    fig3, ev3 = plot_3d(time_val)
    placeholder2d.pyplot(fig2)
    placeholder3d.pyplot(fig3)
    events = ev2 + ev3
    events_placeholder.subheader("Events")
    events_placeholder.write("\n".join(events) or "No events.")

if play:
    for t0 in range(t, int(df.time.max())+1):
        render_frame(t0)
        time.sleep(delay)
else:
    render_frame(t)
