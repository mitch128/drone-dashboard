import streamlit as st
import pandas as pd
import numpy as np
import math
import time
from datetime import timedelta
import pydeck as pdk
import plotly.graph_objects as go

# --- Realistic Dummy Data Generation with Simulated YOLO Detections ---
@st.cache_data
def make_dummy_data(duration=120, fps=1, cam_fov_deg=90, img_size=(1280, 720)):
    """
    Simulate stereo camera YOLO detections followed by triangulation to 3D positions.
    Returns:
      - detections: DataFrame of raw camera detections (time, cam_id, bbox, conf, id)
      - tracks_3d: DataFrame of triangulated drone positions (time, id, type, x, y, z)
    """
    np.random.seed(2025)
    times = np.arange(0, duration+1, 1/fps)
    drone_defs = [
        {'id': 'D1', 'type': 'Shahed Missile',
         'start': np.array([-1000., 800., 1000.]), 'target': np.array([200., 100., 0.]), 'speed': 250.0},
        {'id': 'D2', 'type': 'DJI Mavic',
         'center': np.array([150., -200., 50.]), 'radius': 100., 'period': 60.0},
        {'id': 'D3', 'type': 'Recon UAV',
         'center': np.array([-150., -100., 300.]), 'radius': 200., 'period': 180.0},
    ]
    detections = []
    tracks = []

    # stereo camera baseline (meters)
    baseline = 1.0
    focal = img_size[0] / (2 * math.tan(math.radians(cam_fov_deg/2)))

    for t in times:
        true_positions = {}
        # generate true positions for each drone
        for d in drone_defs:
            if d['id'] == 'D1':
                frac = min(t / duration, 1)
                pos = d['start'] * (1-frac) + d['target'] * frac
                # linear altitude drop
                pos[2] = max(d['start'][2]*(1-frac), 0)
                # wind drift noise
                pos[:2] += np.array([np.sin(t/15), np.cos(t/20)]) * 5
            elif d['id'] == 'D2':
                angle = 2*np.pi*(t % d['period'])/d['period']
                pos = d['center'] + np.array([math.cos(angle), math.sin(angle), 0]) * d['radius']
                pos[2] += 10 * math.sin(2*np.pi*t/30)
            else:
                angle = 2*np.pi*(t % d['period'])/d['period']
                xy = d['center'][:2] + np.array([math.cos(angle), math.sin(angle)])*d['radius']
                pos = np.array([xy[0], xy[1], d['center'][2]])
            true_positions[d['id']] = pos

        # simulate stereo detections for each camera
        for cam in ['L', 'R']:
            offset_dir = -baseline/2 if cam=='L' else baseline/2
            for did, pos in true_positions.items():
                # project to image plane
                x_cam = pos[0] + offset_dir
                y_cam = pos[1]
                z_cam = pos[2] + 100  # camera height offset
                # simple pinhole
                u = focal * (x_cam / z_cam) + img_size[0]/2 + np.random.normal(0,5)
                v = focal * (y_cam / z_cam) + img_size[1]/2 + np.random.normal(0,5)
                # bounding box size inversely proportional to distance
                size = np.clip(20000 / z_cam, 30, 200)
                bbox = [u-size/2, v-size/2, u+size/2, v+size/2]
                conf = np.clip(0.5 + (200/z_cam) + np.random.normal(0,0.05), 0, 1)
                detections.append({'time': t, 'cam_id': cam, 'id': did,
                                   'bbox': bbox, 'conf': conf})
        # triangulate between L and R
        for did in true_positions:
            detL = detections[-2*len(drone_defs) + list(true_positions).index(did)]
            detR = detections[-len(drone_defs) + list(true_positions).index(did)]
            # compute disparity
            uL = np.mean(detL['bbox'][[0,2]])
            uR = np.mean(detR['bbox'][[0,2]])
            disp = (uL - uR)
            if abs(disp) < 1e-2: disp = 1e-2
            Z = focal * baseline / disp
            X = (uL - img_size[0]/2) * Z / focal
            Y = (detL['bbox'][1] + detL['bbox'][3])/2 - img_size[1]/2
            Y = Y * Z / focal
            # add triangulation noise
            pos3d = np.array([X, Y, Z-100]) + np.random.normal(0,2,3)
            tracks.append({'time': t, 'id': did,
                           'type': next(d['type'] for d in drone_defs if d['id']==did),
                           'x': pos3d[0], 'y': pos3d[1], 'z': pos3d[2]})

    return pd.DataFrame(detections), pd.DataFrame(tracks)

# Generate data
detections_df, df = make_dummy_data(duration=120, fps=2)

# Infantry positions (ground reference)
infantry = {
    'Alpha HQ': (0, 0, 0),
    'Bravo FOB': (200, 100, 0),
    'Charlie OP': (-150, -100, 0)
}

def compute_impact(row, prev_row):
    v = np.array([row.x - prev_row.x, row.y - prev_row.y, row.z - prev_row.z])
    if np.linalg.norm(v) < 1:
        return None
    dz = row.z
    t_fall = math.sqrt(2*dz/9.81)
    return np.array([row.x + v[0]*t_fall, row.y + v[1]*t_fall, 0])

# Streamlit App
st.set_page_config(page_title="Drone Dashboard", layout="wide")
st.title("🚁 Drone Tracking & Threat Visualization")

duration = int(df.time.max())
current_t = st.sidebar.slider("Time (s)", 0, duration, 0)
play = st.sidebar.button("▶️ Play")
speed = st.sidebar.slider("Speed (fps)", 1, 5, 2)

# Display raw detections
st.sidebar.subheader("Sample Detections")
st.sidebar.dataframe(detections_df[detections_df.time==current_t].drop('time', axis=1))

# 2D Radar View
tab2d, tab3d, tab_map = st.tabs(["2D Radar", "3D View", "Map View"])

with tab2d:
    fig2d = go.Figure()
    for r in [100,300,600]:
        fig2d.add_shape(type="circle", x0=-r,y0=-r,x1=r,y1=r,
                        line=dict(dash='dash',color='gray'))
    
    for name,pos in infantry.items():
        fig2d.add_trace(go.Scatter(x=[pos[0]], y=[pos[1]], mode='markers+text',
                                   marker=dict(symbol='square',size=12), text=[name]))
    df_view = df[df.time<=current_t]
    for did, g in df_view.groupby('id'):
        fig2d.add_trace(go.Scatter(x=g.x, y=g.y, mode='lines', name=did))
    
    st.plotly_chart(fig2d, use_container_width=True)

with tab3d:
    df_now = df[df.time==current_t]
    fig3d = go.Figure()
    for _,r in df_now.iterrows():
        fig3d.add_trace(go.Scatter3d(x=[r.x],y=[r.y],z=[r.z],mode='markers+text', text=[r.id]))
    st.plotly_chart(fig3d, use_container_width=True)

with tab_map:
    df_now = df[df.time==current_t]
    midpoint = (df_now.y.mean(), df_now.x.mean())
    deck = pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=10),
        layers=[pdk.Layer('ScatterplotLayer',
                          data=df_now.rename(columns={'x':'lon','y':'lat'}),
                          get_position=['lon','lat'], get_radius=50)])
    st.pydeck_chart(deck)

# Playback
if play:
    for t in range(current_t, duration+1, int(1/speed)):
        time.sleep(1/speed)
        st.experimental_rerun()
