import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import pydeck as pdk

def make_dummy_data(duration=120, dt=1):
    """
    Simulate 3 drones with distinct motion patterns in 3D.
    Returns DataFrame with columns: time, id, type, x, y, z
    """
    times = np.arange(0, duration + dt, dt)
    drones = [
        { 'id':'D1', 'type':'Shahed Missile', 'start':np.array([-1000.,800.,1000.]), 'target':np.array([200.,100.,0.]) },
        { 'id':'D2', 'type':'DJI Mavic',     'center':np.array([150.,-200.,50.]), 'radius':100., 'period':60.0 },
        { 'id':'D3', 'type':'Recon UAV',     'center':np.array([-150.,-100.,300.]), 'radius':200., 'period':180.0 }
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
                if d['id']=='D2':
                    xy = d['center'][:2] + np.array([math.cos(angle),math.sin(angle)])*d['radius']
                    z  = d['center'][2] + 10*math.sin(2*math.pi*t/30)
                else:
                    xy = d['center'][:2] + np.array([math.cos(angle),math.sin(angle)])*d['radius']
                    z  = d['center'][2]
                pos = np.array([xy[0], xy[1], z])
            records.append({'time':int(t), 'id':d['id'], 'type':d['type'],
                            'x':pos[0], 'y':pos[1], 'z':pos[2]})
    return pd.DataFrame(records)

# Generate dummy tracks
tracks_df = make_dummy_data(duration=120, dt=1)

# Infantry positions (ground)
infantry = {
    'Alpha HQ': (0, 0, 0),
    'Bravo FOB': (200, 100, 0),
    'Charlie OP': (-150, -100, 0)
}

def euclid(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def generate_summary(frame):
    # Counts
    counts = frame['type'].value_counts().to_dict()
    lines = ["**Current Drone Counts:**"]
    for t, c in counts.items(): lines.append(f"- {t}: {c}")
    # Nearest drone per infantry
    lines.append("\n**Nearest Drone per Unit:**")
    for name, pos in infantry.items():
        dmin, closest = float('inf'), None
        for _, r in frame.iterrows():
            dist = euclid(pos, (r.x, r.y, r.z))
            if dist < dmin:
                dmin, closest = dist, f"{r['type']} ({r['id']})"
        if closest:
            lines.append(f"- {name}: {closest} at {dmin:.1f} m")
        else:
            lines.append(f"- {name}: No drone detected")
    return "\n".join(lines)

# Streamlit app
st.set_page_config(page_title="Drone Intelligence Dashboard", layout="wide")
st.title("🚁 Drone Tracking & Threat Visualization")

# Sidebar controls
max_time = int(tracks_df.time.max())
current_t = st.sidebar.slider("Time (s)", 0, max_time, 0)

# Summary and events
st.sidebar.subheader("Battlefield Summary")
frame_now = tracks_df[tracks_df.time==current_t]
st.sidebar.markdown(generate_summary(frame_now))

# Tabs for views
tab2d, tab3d, tabmap = st.tabs(["2D Radar", "3D View", "Map View"])

# 2D Radar plot
with tab2d:
    fig2d = go.Figure()
    # rings
    for r in [100, 300, 600]:
        fig2d.add_shape(type="circle", x0=-r, y0=-r, x1=r, y1=r,
                        line=dict(dash='dash', color='gray'))
    # infantry
    for name, pos in infantry.items():
        fig2d.add_trace(go.Scatter(x=[pos[0]], y=[pos[1]], mode='markers+text',
                                   marker=dict(symbol='square', size=12),
                                   text=[name], textposition='top right', showlegend=False))
    # past tracks
    df_hist = tracks_df[tracks_df.time<=current_t]
    for did, grp in df_hist.groupby('id'):
        fig2d.add_trace(go.Scatter(x=grp.x, y=grp.y, mode='lines', name=did))
    # current positions
    for _, r in frame_now.iterrows():
        color = 'red' if 'Shahed' in r.type else ('blue' if 'Mavic' in r.type else 'green')
        fig2d.add_trace(go.Scatter(x=[r.x], y=[r.y], mode='markers+text',
                                   marker=dict(size=14, color=color),
                                   text=[r.id], textposition='bottom right', showlegend=False))
    fig2d.update_layout(xaxis=dict(range=[-1200,1200]), yaxis=dict(range=[-1200,1200]),
                        title=f"2D Radar at t={current_t}s", height=600)
    st.plotly_chart(fig2d, use_container_width=True)

# 3D View plot
with tab3d:
    fig3d = go.Figure()
    for _, r in frame_now.iterrows():
        fig3d.add_trace(go.Scatter3d(x=[r.x], y=[r.y], z=[r.z], mode='markers+text',
                                    marker=dict(size=6), text=[r.id], textposition='bottom right'))
    fig3d.update_layout(scene=dict(xaxis=dict(range=[-1200,1200]),
                                   yaxis=dict(range=[-1200,1200]),
                                   zaxis=dict(range=[0,1200])),
                        title=f"3D View at t={current_t}s", height=600)
    st.plotly_chart(fig3d, use_container_width=True)

# Map view with pydeck
with tabmap:
    df_map = frame_now.rename(columns={'x':'lon','y':'lat'})
    layer = pdk.Layer('ScatterplotLayer', data=df_map,
                      get_position='[lon, lat]', get_radius=50, pickable=True)
    view = pdk.ViewState(longitude=df_map.lon.mean(), latitude=df_map.lat.mean(), zoom=10)
    deck = pdk.Deck(layers=[layer], initial_view_state=view)
    st.pydeck_chart(deck)
