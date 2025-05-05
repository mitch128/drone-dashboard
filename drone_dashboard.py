import streamlit as st
st.set_page_config(page_title="Drone Dashboard", layout="wide")

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
    times = np.linspace(0, duration, int(duration * fps) + 1)
    drone_defs = [
        {'id': 'D1', 'type': 'Shahed Missile',
         'start': np.array([-1000., 800., 1000.]), 'target': np.array([200., 100., 0.])},
        {'id': 'D2', 'type': 'DJI Mavic',
         'center': np.array([150., -200., 50.]), 'radius': 100., 'period': 60.0},
        {'id': 'D3', 'type': 'Recon UAV',
         'center': np.array([-150., -100., 300.]), 'radius': 200., 'period': 180.0},
    ]
    detections, tracks = [], []
    baseline = 1.0  # meters
    focal = img_size[0] / (2 * math.tan(math.radians(cam_fov_deg / 2)))

    for t in times:
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

        # YOLO detections
        for cam in ['L', 'R']:
            offset = -baseline / 2 if cam == 'L' else baseline / 2
            for did, pos in true_positions.items():
                x_cam, y_cam = pos[0] + offset, pos[1]
                z_cam = pos[2] + 100
                u = focal * (x_cam / z_cam) + img_size[0] / 2 + np.random.normal(0, 5)
                v = focal * (y_cam / z_cam) + img_size[1] / 2 + np.random.normal(0, 5)
                size = np.clip(20000 / z_cam, 30, 200)
                bbox = [u - size/2, v - size/2, u + size/2, v + size/2]
                conf = float(np.clip(0.5 + (200 / z_cam) + np.random.normal(0, 0.05), 0, 1))
                detections.append({'time': t, 'cam_id': cam, 'id': did, 'bbox': bbox, 'conf': conf})
        # Triangulation
        for i, did in enumerate(true_positions):
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
            tracks.append({'time': t, 'id': did, 'type': next(d['type'] for d in drone_defs if d['id'] == did),
                           'x': pos3d[0], 'y': pos3d[1], 'z': pos3d[2]})

    detections_df = pd.DataFrame(detections)
    tracks_df = pd.DataFrame(tracks)
    # Round times to int
    detections_df['time'] = detections_df['time'].round().astype(int)
    tracks_df['time'] = tracks_df['time'].round().astype(int)
    return detections_df, tracks_df

# Generate data
# detections_df: raw YOLO outputs, df: 3D tracks

detections_df, df = make_dummy_data(duration=120, fps=2)

# Impact projection
def compute_impact(row, prev_row):
    v = np.array([row.x - prev_row.x, row.y - prev_row.y, row.z - prev_row.z])
    if np.linalg.norm(v) < 1:
        return None
    dz = row.z
    t_fall = math.sqrt(2 * dz / 9.81)
    return np.array([row.x + v[0] * t_fall, row.y + v[1] * t_fall, 0])

# Infantry ground positions
infantry = {'Alpha HQ': (0, 0, 0), 'Bravo FOB': (200, 100, 0), 'Charlie OP': (-150, -100, 0)}

# App UI
st.title("🚁 Drone Tracking & Threat Visualization")
# Controls
duration = int(df.time.max())
current_t = st.sidebar.slider("Time (s)", 0, duration, 0)

# Show sample detections
st.sidebar.subheader("YOLO Detections at t")
st.sidebar.dataframe(detections_df[detections_df.time == current_t].drop(columns=['time']))

# Views
tab2d, tab3d, tab_map = st.tabs(["2D Radar", "3D View", "Map View"])

# 2D Radar
def render_2d(time_t):
    fig = go.Figure()
    for r in [100, 300, 600]:
        fig.add_shape(type="circle", x0=-r, y0=-r, x1=r, y1=r, line=dict(dash='dash', color='gray'))
    for name, pos in infantry.items():
        fig.add_trace(go.Scatter(x=[pos[0]], y=[pos[1]], mode='markers+text', marker=dict(symbol='square', size=12), text=[name], textposition='top right'))
    df_slice = df[df.time <= time_t]
    for did, grp in df_slice.groupby('id'):
        fig.add_trace(go.Scatter(x=grp.x, y=grp.y, mode='lines', name=did))
    fig.update_layout(xaxis=dict(range=[-800, 800]), yaxis=dict(range=[-800, 800]), title=f"2D Radar at t={time_t}s")
    return fig

# 3D View
def render_3d(time_t):
    df_now = df[df.time == time_t]
    fig = go.Figure()
    for _, r in df_now.iterrows():
        fig.add_trace(go.Scatter3d(x=[r.x], y=[r.y], z=[r.z], mode='markers+text', text=[r.id]))
    fig.update_layout(scene=dict(xaxis=dict(range=[-800,800]), yaxis=dict(range=[-800,800]), zaxis=dict(range=[0,1200])), title=f"3D View at t={time_t}s")
    return fig

# Map View
def render_map(time_t):
    df_now = df[df.time == time_t]
    midpoint = (df_now.y.mean(), df_now.x.mean())
    return pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=10),
        layers=[pdk.Layer('ScatterplotLayer', data=df_now.rename(columns={'x':'lon','y':'lat'}), get_position=['lon','lat'], get_radius=50)]
    )

with tab2d:
    st.plotly_chart(render_2d(current_t), use_container_width=True)
with tab3d:
    st.plotly_chart(render_3d(current_t), use_container_width=True)
with tab_map:
    st.pydeck_chart(render_map(current_t))
