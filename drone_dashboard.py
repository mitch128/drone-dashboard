import streamlit as st
import pandas as pd
import numpy as np
import math
import time
import pydeck as pdk
import plotly.graph_objects as go

# --- Realistic Dummy Data Generation with Simulated YOLO Detections ---
@st.cache_data
def make_dummy_data(duration=120, fps=2, cam_fov_deg=90, img_size=(1280, 720)):
    """
    Simulate stereo camera YOLO detections and triangulate to 3D positions.
    Returns:
      detections_df: DataFrame of raw camera detections
      tracks_df: DataFrame of 3D drone tracks
    """
    np.random.seed(2025)
    times = np.arange(0, duration + 1/fps, 1/fps)
    # Drone definitions
    drone_defs = [
        {'id': 'D1', 'type': 'Shahed Missile',
         'start': np.array([-1000., 800., 1000.]), 'target': np.array([200., 100., 0.])},
        {'id': 'D2', 'type': 'DJI Mavic',
         'center': np.array([150., -200., 50.]), 'radius': 100., 'period': 60.0},
        {'id': 'D3', 'type': 'Recon UAV',
         'center': np.array([-150., -100., 300.]), 'radius': 200., 'period': 180.0},
    ]
    detections = []
    tracks = []

    # Stereo camera parameters
    baseline = 1.0  # meters
    focal = img_size[0] / (2 * math.tan(math.radians(cam_fov_deg / 2)))

    for t in times:
        # Compute true positions
        true_positions = {}
        for d in drone_defs:
            if d['id'] == 'D1':
                frac = min(t / duration, 1)
                pos = d['start'] * (1 - frac) + d['target'] * frac
                pos[2] = max(d['start'][2] * (1 - frac), 0)
                pos[:2] += np.array([np.sin(t / 15), np.cos(t / 20)]) * 5
            elif d['id'] == 'D2':
                angle = 2 * np.pi * (t % d['period']) / d['period']
                pos = d['center'] + np.array([math.cos(angle), math.sin(angle), 0]) * d['radius']
                pos[2] = d['center'][2] + 10 * math.sin(2 * np.pi * t / 30)
            else:
                angle = 2 * np.pi * (t % d['period']) / d['period']
                xy = d['center'][:2] + np.array([math.cos(angle), math.sin(angle)]) * d['radius']
                pos = np.array([xy[0], xy[1], d['center'][2]])
            true_positions[d['id']] = pos

        # Simulate YOLO detections for L and R cameras
        for cam in ['L', 'R']:
            offset = -baseline / 2 if cam == 'L' else baseline / 2
            for did, pos in true_positions.items():
                x_cam = pos[0] + offset
                y_cam = pos[1]
                z_cam = pos[2] + 100  # camera height
                u = focal * (x_cam / z_cam) + img_size[0] / 2 + np.random.normal(0, 5)
                v = focal * (y_cam / z_cam) + img_size[1] / 2 + np.random.normal(0, 5)
                size = np.clip(20000 / z_cam, 30, 200)
                bbox = [u - size / 2, v - size / 2, u + size / 2, v + size / 2]
                conf = float(np.clip(0.5 + (200 / z_cam) + np.random.normal(0, 0.05), 0, 1))
                detections.append({'time': t, 'cam_id': cam, 'id': did,
                                   'bbox': bbox, 'conf': conf})
        # Triangulate detections into 3D
        for i, (did, pos) in enumerate(true_positions.items()):
            detL = detections[-2 * len(drone_defs) + i]
            detR = detections[-len(drone_defs) + i]
            uL = (detL['bbox'][0] + detL['bbox'][2]) / 2
            uR = (detR['bbox'][0] + detR['bbox'][2]) / 2
            disp = uL - uR if abs(uL - uR) > 1e-2 else 1e-2
            Z = focal * baseline / disp
            X = (uL - img_size[0] / 2) * Z / focal
            v_center = (detL['bbox'][1] + detL['bbox'][3]) / 2
            Y = (v_center - img_size[1] / 2) * Z / focal
            pos3d = np.array([X, Y, Z - 100]) + np.random.normal(0, 2, 3)
            tracks.append({'time': t, 'id': did,
                           'type': next(d['type'] for d in drone_defs if d['id'] == did),
                           'x': pos3d[0], 'y': pos3d[1], 'z': pos3d[2]})

    detections_df = pd.DataFrame(detections)
    tracks_df = pd.DataFrame(tracks)
    return detections_df, tracks_df

# Generate data
nd = make_dummy_data(duration=120, fps=2)
detections_df, df = nd

# Ground positions for infantry
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
    t_fall = math.sqrt(2 * dz / 9.81)
    return np.array([row.x + v[0] * t_fall, row.y + v[1] * t_fall, 0])

# Streamlit App Setup
st.set_page_config(page_title="Drone Dashboard", layout="wide")
st.title("🚁 Drone Tracking & Threat Visualization")

# Controls
duration = int(df.time.max())
current_t = st.sidebar.slider("Time (s)", 0, duration, 0)
play = st.sidebar.button("▶️ Play")
speed = st.sidebar.slider("Playback Speed (fps)", 1, 5, 2)

# Show detections table
st.sidebar.subheader("YOLO Detections (sample)")
st.sidebar.dataframe(detections_df[detections_df.time == current_t].drop(columns=['time']))

# Tabs for views
tab2d, tab3d, tab_map = st.tabs(["2D Radar", "3D View", "Map View"])

with tab2d:
    fig2d = go.Figure()
    # Concentric rings
    for r in [100, 300, 600]:
        fig2d.add_shape(type="circle", x0=-r, y0=-r, x1=r, y1=r,
                        line=dict(dash='dash', color='gray'))
    # Infantry markers
    for name, pos in infantry.items():
        fig2d.add_trace(go.Scatter(x=[pos[0]], y=[pos[1]], mode='markers+text',
                                   marker=dict(symbol='square', size=12, color='black'),
                                   text=[name], textposition='top right'))
    # Drone paths
    df_view = df[df.time <= current_t]
    for did, g in df_view.groupby('id'):
        fig2d.add_trace(go.Scatter(x=g.x, y=g.y, mode='lines', name=did))
    fig2d.update_layout(xaxis=dict(range=[-800, 800]),
                        yaxis=dict(range=[-800, 800]),
                        title=f"2D Radar at t={current_t}s")
    st.plotly_chart(fig2d, use_container_width=True)

with tab3d:
    df_now = df[df.time == current_t]
    fig3d = go.Figure()
    for _, r in df_now.iterrows():
        fig3d.add_trace(go.Scatter3d(x=[r.x], y=[r.y], z=[r.z],
                                     mode='markers+text', text=[r.id]))
    fig3d.update_layout(scene=dict(xaxis=dict(range=[-800,800]),
                                   yaxis=dict(range=[-800,800]),
                                   zaxis=dict(range=[0,1200])),
                        title=f"3D View at t={current_t}s")
    st.plotly_chart(fig3d, use_container_width=True)

with tab_map:
    df_now = df[df.time == current_t]
    midpoint = (df_now.y.mean(), df_now.x.mean())
    deck = pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=10),
        layers=[pdk.Layer('ScatterplotLayer', data=df_now.rename(columns={'x':'lon','y':'lat'}),
                          get_position=['lon','lat'], get_radius=50)])
    st.pydeck_chart(deck)

# Auto-play
if play:
    for t in range(current_t, duration + 1, 1):
        time.sleep(1 / speed)
        st.experimental_rerun()
