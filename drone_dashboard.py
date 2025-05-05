import streamlit as st
st.set_page_config(page_title="Drone Dashboard", layout="wide")

import pandas as pd
import numpy as np
import math
import pydeck as pdk
import plotly.graph_objects as go

# --- Realistic Dummy Data Generation with Simulated YOLO Detections ---
@st.cache_data
def make_dummy_data(duration=120, fps=2, cam_fov_deg=90, img_size=(1280, 720)):
    np.random.seed(2025)
    times = np.arange(0, duration + 1, 1, dtype=int)
    drone_defs = [
        {'id': 'D1', 'type': 'Shahed Missile',
         'start': np.array([-1000., 800., 1000.]), 'target': np.array([200., 100., 0.])},
        {'id': 'D2', 'type': 'DJI Mavic',
         'center': np.array([150., -200., 50.]), 'radius': 100., 'period': 60.0},
        {'id': 'D3', 'type': 'Recon UAV',
         'center': np.array([-150., -100., 300.]), 'radius': 200., 'period': 180.0},
    ]
    detections, tracks = [], []
    baseline = 1.0
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

        # YOLO detections for each camera
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
        for idx, did in enumerate(true_positions):
            detL = detections[-2 * len(drone_defs) + idx]
            detR = detections[-len(drone_defs) + idx]
            uL = (detL['bbox'][0] + detL['bbox'][2]) / 2
            uR = (detR['bbox'][0] + detR['bbox'][2]) / 2
            disp = max(abs(uL - uR), 1e-2)
            Z = focal * baseline / disp
            X = (uL - img_size[0] / 2) * Z / focal
            v_center = (detL['bbox'][1] + detL['bbox'][3]) / 2
            Y = (v_center - img_size[1] / 2) * Z / focal
            pos3d = np.array([X, Y, Z - 100]) + np.random.normal(0, 2, 3)
            tracks.append({'time': t, 'id': did, 'type': next(d['type'] for d in drone_defs if d['id'] == did),
                           'x': pos3d[0], 'y': pos3d[1], 'z': pos3d[2]})

    returns = pd.DataFrame(detections), pd.DataFrame(tracks)
    return returns

# Generate data

# Use clear variable names
detections_df, tracks_df = make_dummy_data(duration=120, fps=2)

# Infantry ground positions
infantry = {'Alpha HQ': (0, 0, 0), 'Bravo FOB': (200, 100, 0), 'Charlie OP': (-150, -100, 0)}

# App UI
st.title("🚁 Drone Tracking & Threat Visualization")

# Controls
max_time = int(tracks_df.time.max())
current_t = st.sidebar.slider("Time (s)", 0, max_time, 0)

# Show YOLO detections
st.sidebar.subheader("YOLO Detections at t")
st.sidebar.dataframe(
    detections_df[detections_df.time == current_t]
        .drop(columns=['time'])
        .reset_index(drop=True)
)

# Tabs for different views
tab2d, tab3d, tab_map = st.tabs(["2D Radar", "3D View", "Map View"])

# 2D Radar view
def render_2d(time_t):
    fig = go.Figure()
    for radius in [100, 300, 600]:
        fig.add_shape(type="circle", x0=-radius, y0=-radius, x1=radius, y1=radius,
                      line=dict(dash='dash', color='gray'))
    # Add infantry positions
    for name, pos in infantry.items():
        fig.add_trace(go.Scatter(x=[pos[0]], y=[pos[1]], mode='markers+text',
                                 marker=dict(symbol='square', size=12),
                                 text=[name], textposition='top right'))
    # Add drone tracks
    df_slice = tracks_df[tracks_df.time <= time_t]
    for did, grp in df_slice.groupby('id'):
        fig.add_trace(go.Scatter(x=grp.x, y=grp.y, mode='lines', name=did))
    fig.update_layout(
        xaxis=dict(range=[-1200, 1200]),
        yaxis=dict(range=[-1200, 1200]),
        title=f"2D Radar at t={time_t}s",
    )
    return fig

# 3D View
def render_3d(time_t):
    df_now = tracks_df[tracks_df.time == time_t]
    fig = go.Figure()
    for _, r in df_now.iterrows():
        fig.add_trace(go.Scatter3d(x=[r.x], y=[r.y], z=[r.z], mode='markers+text', text=[r.id]))
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1200,1200]),
            yaxis=dict(range=[-1200,1200]),
            zaxis=dict(range=[0,1200])
        ),
        title=f"3D View at t={time_t}s"
    )
    return fig

# Map View (using pydeck scatterplot)
def render_map(time_t):
    df_now = tracks_df[tracks_df.time == time_t]
    # Treat x,y as lon,lat for demonstration purposes
    df_map = df_now.rename(columns={'x':'lon','y':'lat'})
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=df_map,
        get_position='[lon, lat]',
        get_radius=50,
        pickable=True,
    )
    view_state = pdk.ViewState(
        longitude=df_map.lon.mean(), latitude=df_map.lat.mean(), zoom=10
    )
    return pdk.Deck(layers=[layer], initial_view_state=view_state)

with tab2d:
    st.plotly_chart(render_2d(current_t), use_container_width=True)
with tab3d:
    st.plotly_chart(render_3d(current_t), use_container_width=True)
with tab_map:
    st.pydeck_chart(render_map(current_t))

# End of app
