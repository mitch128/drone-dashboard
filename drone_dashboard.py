import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import pydeck as pdk

# --- Dummy Data Generation ---
def make_dummy_data(duration=120, dt=1):
    times = np.arange(0, duration + dt, dt)
    drones = [
        {'id':'D1','type':'Shahed Missile','start':np.array([-1000.,800.,1000.]),'target':np.array([200.,100.,0.])},
        {'id':'D2','type':'DJI Mavic','center':np.array([150.,-200.,50.]),'radius':100.,'period':60.0},
        {'id':'D3','type':'Recon UAV','center':np.array([-150.,-100.,300.]),'radius':200.,'period':180.0}
    ]
    records = []
    for t in times:
        for d in drones:
            if d['id']=='D1':
                frac = min(t/duration,1)
                pos = d['start']*(1-frac) + d['target']*frac
                pos += np.array([np.sin(t/15), np.cos(t/20), 0])*5
            else:
                angle = 2*math.pi*(t % d['period'])/d['period']
                xy = d['center'][:2] + np.array([math.cos(angle),math.sin(angle)])*d['radius']
                z = d['center'][2] + (10*math.sin(2*math.pi*t/30) if d['id']=='D2' else 0)
                pos = np.array([xy[0], xy[1], z])
            records.append({'time':int(t),'id':d['id'],'type':d['type'],'x':pos[0],'y':pos[1],'z':pos[2]})
    return pd.DataFrame(records)

# Generate data
df = make_dummy_data(duration=120, dt=1)

# Infantry positions (x,y,z)
infantry = {'Alpha HQ':(0,0,0),'Bravo FOB':(200,100,0),'Charlie OP':(-150,-100,0)}

def euclid(a,b): return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def generate_summary(frame):
    counts = frame['type'].value_counts().to_dict()
    lines = ["**Current Drone Counts:**"]
    for t,c in counts.items(): lines.append(f"- {t}: {c}")
    lines.append("\n**Nearest Drone per Unit:**")
    for name,pos in infantry.items():
        dmin,closest=float('inf'),None
        for _,r in frame.iterrows():
            dist=euclid(pos,(r.x,r.y,r.z))
            if dist<dmin: dmin,closest=dist,f"{r['type']} ({r['id']})"
        if closest: lines.append(f"- {name}: {closest} at {dmin:.1f} m")
        else: lines.append(f"- {name}: No drone detected")
    return "\n".join(lines)

# Streamlit setup
st.set_page_config(page_title="Drone Intelligence Dashboard", layout="wide")
st.title("🚁 Drone Tracking & Threat Visualization")

# Time selector
t_max=int(df.time.max())
t=st.sidebar.slider("Time (s)",0,t_max,0)
# Summary
st.sidebar.subheader("Battlefield Summary")
frame_now=df[df.time==t]
st.sidebar.markdown(generate_summary(frame_now))

# Columns: give a bit more width to plots
col2d, col3d = st.columns([1.2,1])

# 2D Radar
with col2d:
    st.subheader(f"2D Radar (t={t}s)")
    fig2d=go.Figure()
    # rings
    for r in [100,300,600]:
        fig2d.add_shape(type='circle',x0=-r,y0=-r,x1=r,y1=r, line=dict(dash='dash',color='gray'))
    # infantry
    for name,pos in infantry.items():
        fig2d.add_trace(go.Scatter(x=[pos[0]],y=[pos[1]],mode='markers+text',
                                   marker=dict(symbol='square',size=14,color='black'),
                                   text=[name],textposition='top center',showlegend=False))
    # drone history
    hist=df[df.time<=t]
    for did,grp in hist.groupby('id'):
        fig2d.add_trace(go.Scatter(x=grp.x,y=grp.y,mode='lines',name=did))
    # current drones + confidence areas
    for _,r in frame_now.iterrows():
        # shading: precise vs coarse
        small,large=50,150
        fig2d.add_shape(type='circle',x0=r.x-small,y0=r.y-small,x1=r.x+small,y1=r.y+small,
                        fillcolor='rgba(135,206,250,0.4)',line_width=0)
        fig2d.add_shape(type='circle',x0=r.x-large,y0=r.y-large,x1=r.x+large,y1=r.y+large,
                        fillcolor='rgba(135,206,250,0.2)',line_width=0)
        colr='red' if 'Shahed' in r.type else ('blue' if 'Mavic' in r.type else 'green')
        fig2d.add_trace(go.Scatter(x=[r.x],y=[r.y],mode='markers+text',
                                   marker=dict(size=16,color=colr),text=[r.id],textposition='bottom right',showlegend=False))
    fig2d.update_layout(xaxis=dict(range=[-1200,1200]),yaxis=dict(range=[-1200,1200]),height=650)
    st.plotly_chart(fig2d,use_container_width=True)

# 3D View
with col3d:
    st.subheader(f"3D View (t={t}s)")
    fig3d=go.Figure()
    hist3d=df[df.time<=t]
    for did,grp in hist3d.groupby('id'):
        fig3d.add_trace(go.Scatter3d(x=grp.x,y=grp.y,z=grp.z,mode='lines',name=did))
    for _,r in frame_now.iterrows():
        # confidence spheres: approximate with translucent markers
        fig3d.add_trace(go.Scatter3d(x=[r.x],y=[r.y],z=[r.z],mode='markers',
                                    marker=dict(size=large/2,opacity=0.2,color='lightblue'),showlegend=False))
        fig3d.add_trace(go.Scatter3d(x=[r.x],y=[r.y],z=[r.z],mode='markers',
                                    marker=dict(size=small/2,opacity=0.4,color='lightblue'),showlegend=False))
        fig3d.add_trace(go.Scatter3d(x=[r.x],y=[r.y],z=[r.z],mode='markers+text',
                                    marker=dict(size=8,color='orange'),text=[r.id],textposition='top center',showlegend=False))
    # infantry bases
    for name,pos in infantry.items():
        fig3d.add_trace(go.Scatter3d(x=[pos[0]],y=[pos[1]],z=[pos[2]],mode='markers+text',
                                    marker=dict(symbol='square',size=10,color='black'),text=[name],textposition='top center',showlegend=False))
    fig3d.update_layout(scene=dict(xaxis=dict(range=[-1200,1200]),yaxis=dict(range=[-1200,1200]),
                                   zaxis=dict(range=[0,1200])),height=650)
    st.plotly_chart(fig3d,use_container_width=True)

# Map View below
st.subheader("Map View of Drones")
fig_map=pdk.Deck(layers=[pdk.Layer('ScatterplotLayer',data=frame_now.rename(columns={'x':'lon','y':'lat'}),
                                 get_position='[lon, lat]',get_radius=50,pickable=True)],
                 initial_view_state=pdk.ViewState(longitude=float(frame_now.x.mean()),
                                                 latitude=float(frame_now.y.mean()),zoom=10))
st.pydeck_chart(fig_map)
