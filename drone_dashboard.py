import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import pydeck as pdk

# --- Dummy Data Generation ---
def make_dummy_data(duration=120, dt=1):
    """
    Simulate 3 drones with distinct motion patterns in 3D.
    Returns DataFrame with columns: time, id, type, x, y, z
    """
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

# Generate dummy tracks
df = make_dummy_data(duration=120, dt=1)

# Infantry positions (ground)
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
        lines.append(f"- {name}: {closest if closest else 'No drone detected'} at {dmin:.1f} m" if closest else f"- {name}: No drone detected")
    return "\n".join(lines)

# Streamlit UI
st.set_page_config(page_title="Drone Intelligence Dashboard", layout="wide")
st.title("🚁 Drone Tracking & Threat Visualization")

# Sidebar controls
t_max = int(df.time.max())
t = st.sidebar.slider("Time (s)", 0, t_max, 0)

# Summary panel
st.sidebar.subheader("Battlefield Summary")
frame_now = df[df.time==t]
st.sidebar.markdown(generate_summary(frame_now))

# Tabs
tab2d,tab3d,tabmap = st.tabs(["2D Radar","3D View","Map View"])

# 2D Radar
with tab2d:
    fig2d = go.Figure()
    for r in [100,300,600]:
        fig2d.add_shape(type='circle', x0=-r,y0=-r,x1=r,y1=r, line=dict(dash='dash',color='gray'))
    for name,pos in infantry.items():
        fig2d.add_trace(go.Scatter(x=[pos[0]],y=[pos[1]],mode='markers+text',marker=dict(symbol='square',size=12),text=[name],textposition='top right',showlegend=False))
    hist=df[df.time<=t]
    for did,grp in hist.groupby('id'):
        fig2d.add_trace(go.Scatter(x=grp.x,y=grp.y,mode='lines',name=did))
    for _,r in frame_now.iterrows():
        color='red' if 'Shahed' in r.type else ('blue' if 'Mavic' in r.type else 'green')
        fig2d.add_trace(go.Scatter(x=[r.x],y=[r.y],mode='markers+text',marker=dict(size=14,color=color),text=[r.id],textposition='bottom right',showlegend=False))
    fig2d.update_layout(xaxis=dict(range=[-1200,1200]),yaxis=dict(range=[-1200,1200]),title=f"2D Radar at t={t}s",height=600)
    st.plotly_chart(fig2d,use_container_width=True)

# 3D View
with tab3d:
    fig3d = go.Figure()
    # show trajectory lines in 3D
    hist3d = df[df.time<=t]
    for did,grp in hist3d.groupby('id'):
        fig3d.add_trace(go.Scatter3d(x=grp.x,y=grp.y,z=grp.z,mode='lines',name=did))
    # show current positions
    for _,r in frame_now.iterrows():
        fig3d.add_trace(go.Scatter3d(x=[r.x],y=[r.y],z=[r.z],mode='markers+text',marker=dict(size=6),text=[r.id]))
    fig3d.update_layout(scene=dict(xaxis=dict(range=[-1200,1200]),yaxis=dict(range=[-1200,1200]),zaxis=dict(range=[0,1200])),title=f"3D View at t={t}s",height=600)
    st.plotly_chart(fig3d,use_container_width=True)

# Map View
with tabmap:
    df_map = frame_now.rename(columns={'x':'lon','y':'lat'})
    layer = pdk.Layer('ScatterplotLayer', data=df_map, get_position='[lon, lat]', get_radius=50, pickable=True)
    view = pdk.ViewState(longitude=float(df_map.lon.mean()),latitude=float(df_map.lat.mean()),zoom=10)
    deck = pdk.Deck(layers=[layer], initial_view_state=view)
    st.pydeck_chart(deck)
