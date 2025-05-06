import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import time
import math

# --- Apply military-style CSS for Streamlit ---
st.markdown("""
<style>
body {background-color: #000000; color: #A3DE83;}
.reportview-container .main .block-container {background-color: #000000;}
.sidebar .sidebar-content {background-color: #1A1A1A; color: #A3DE83;}
h1, h2, h3, .block-container .element-container span {color: #A3DE83;}
.stButton>button, .stSlider>div>div>div>input {background-color: #2E4E1D; color: #A3DE83;}
</style>
""", unsafe_allow_html=True)

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
        noise = np.random.normal(0,2,3)
        pos = start + v0*t + dev + noise
        rows.append({'time':t, 'id':'D1','x':pos[0],'y':pos[1],'z':pos[2]})
    # D2 less-circular
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
    # velocities
    for d in df['id'].unique():
        m = df['id']==d
        df.loc[m, ['vx','vy','vz']] = df[m][['x','y','z']].diff().fillna(0)
    return df

df = make_dummy_data()

# Friendly unit
infantry_positions = {"Central Unit": (0,0,0)}

# Impact zone util
def compute_impact(x,y,z,vx,vy,vz,sec=5):
    px,py,pz = x+vx*sec, y+vy*sec, z+vz*sec
    dist = math.sqrt((vx*sec)**2+(vy*sec)**2+(vz*sec)**2)
    r = 0.15*dist+15
    return px,py,pz,r

# sphere sizes
sizes = {'D1':50,'D2':40,'D3':30}

#######################################
# 3D Plot
#######################################
def plot_3d(t):
    fig = plt.figure(facecolor='black', figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    ax.set_xlim(-700,700); ax.set_ylim(-700,700); ax.set_zlim(0,400)
    ax.w_xaxis.set_pane_color((0,0,0,1)); ax.w_yaxis.set_pane_color((0,0,0,1)); ax.w_zaxis.set_pane_color((0,0,0,1))
    ax.set_title(f"Time = {t}s", color='#A3DE83')
    # ground
    xx,yy = np.meshgrid(np.linspace(-700,700,2), np.linspace(-700,700,2))
    ax.plot_surface(xx,yy,np.zeros_like(xx),color='#2E4E1D',alpha=0.3)
    # friendly
    for pos in infantry_positions.values(): ax.scatter(*pos, c='#A3DE83', marker='^', s=150)
    # history
    for d in df['id'].unique(): hist=df[(df.id==d)&(df.time<=t)]; ax.plot(hist.x,hist.y,hist.z, linestyle='dotted', color='gray', alpha=0.5)
    events=[]
    for _,r in df[df.time==t].iterrows():
        c = '#FF4136' if r.id=='D1' else '#0074D9' if r.id=='D2' else '#2ECC40'
        rsz=sizes[r.id]
        # sphere
        u,v = np.mgrid[0:2*np.pi:12j,0:np.pi:6j]
        xs = r.x + rsz*np.cos(u)*np.sin(v); ys = r.y + rsz*np.sin(u)*np.sin(v); zs = r.z + rsz*np.cos(v)
        ax.plot_surface(xs,ys,zs,color=c,alpha=0.2)
        # marker half size
        ax.scatter(r.x,r.y,r.z,c=c,s=20)
        ax.text(r.x+10,r.y+10,r.z+10, r.id, color=c)
        if r.id=='D1': px,py,pz,rad = compute_impact(r.x,r.y,r.z,r.vx,r.vy,r.vz); events.append(f"D1 impact at ({px:.0f},{py:.0f},{pz:.0f}) r={rad:.0f}m")
    return fig, events

#######################################
# 2D Plot
#######################################
def plot_2d(t):
    fig,ax = plt.subplots(facecolor='black', figsize=(8,8))
    ax.set_facecolor('black'); ax.set_xlim(-700,700); ax.set_ylim(-700,700)
    ax.set_title(f"Time = {t}s", color='#A3DE83')
    for pos in infantry_positions.values(): ax.scatter(pos[0],pos[1],c='#A3DE83',marker='^',s=100)
    for d in df['id'].unique(): hist=df[(df.id==d)&(df.time<=t)]; ax.plot(hist.x,hist.y, linestyle='dotted', color='gray', alpha=0.5)
    events=[]
    for _,r in df[df.time==t].iterrows():
        c = '#FF4136' if r.id=='D1' else '#0074D9' if r.id=='D2' else '#2ECC40'
        rsz=sizes[r.id]
        ax.add_patch(plt.Circle((r.x,r.y), rsz, color=c, alpha=0.2))
        ax.plot(r.x,r.y,'o',color=c,markersize=2)
        ax.text(r.x+10,r.y+10, r.id, color=c)
        if r.id=='D1': px,py,_,rad=compute_impact(r.x,r.y,r.z,r.vx,r.vy,r.vz); events.append(f"D1 2D impact at ({px:.0f},{py:.0f}) r={rad:.0f}m")
    ax.grid(True, color='gray', linestyle='--', alpha=0.3)
    return fig, events

# --- Streamlit UI ---
st.set_page_config(page_title="Drone Dashboard", layout="wide")
st.title("Drone Intelligence Dashboard")

# Controls
with st.sidebar:
    st.header("Controls")
    t = st.slider("Time (s)", 0, int(df.time.max()), 0)
    play = st.button("Play")
    delay = st.number_input("Speed s/frame", 0.1,5.0,0.75)

# Layout
col1,col2 = st.columns(2)
events_log = []
if play:
    for t0 in range(t, int(df.time.max())+1):
        fig2, ev2 = plot_2d(t0); fig3, ev3 = plot_3d(t0)
        col1.pyplot(fig2); col2.pyplot(fig3)
        events_log = ev2+ev3
        st.subheader("Events")
        st.write("\n".join(events_log) or "No events.")
        time.sleep(delay)
else:
    fig2, ev2 = plot_2d(t); fig3, ev3 = plot_3d(t)
    col1.pyplot(fig2); col2.pyplot(fig3)
    events_log = ev2+ev3
    st.subheader("Events")
    st.write("\n".join(events_log) or "No events.")
