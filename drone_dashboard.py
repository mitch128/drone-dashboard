import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import math

#-----------------------------------------------------------
# 1. Define Simulation Parameters & Geographic Helpers
#-----------------------------------------------------------
reference_locations = {
    "Kursk, Russia": {"lat": 51.7300, "lon": 36.1900},
    "Odessa, Ukraine": {"lat": 46.4825, "lon": 30.7233},
    "Kiev, Ukraine": {"lat": 50.4501, "lon": 30.5234},
}

def to_geo(x, y, ref_lat, ref_lon):
    # Use a simple conversion: 1 degree latitude ~111 km; adjusted for longitude.
    delta_lat = x / 111000  # meters to degrees latitude
    delta_lon = y / (111000 * np.cos(np.deg2rad(ref_lat)))
    return ref_lat + delta_lat, ref_lon + delta_lon

#-----------------------------------------------------------
# 2. Generate Realistic Dummy Drone Data (Local Coords in meters)
#-----------------------------------------------------------
def make_dummy_data():
    np.random.seed(42)
    timestamps = np.arange(0, 31)  # simulate 31 seconds
    rows = []
    
    # Drone D1: High-threat (Shahed) with a smooth curved descent.
    start_D1 = np.array([-600, -500, 200])
    base_velocity_D1 = np.array([30, 35, -4])
    for t in timestamps:
        deviation = np.array([20 * np.sin(0.08*t), 20 * np.cos(0.08*t), 5 * np.sin(0.04*t)])
        noise = np.random.normal(0, 2, 3)
        pos = start_D1 + base_velocity_D1 * t + deviation + noise
        rows.append({
            'time': t,
            'id': 'D1',
            'type': 'Shahed',
            'x': pos[0],
            'y': pos[1],
            'z': pos[2]
        })
    
    # Drone D2: Recon drone (DJI Mavic) following a looping spiral.
    start_D2 = np.array([200, 400, 100])
    for t in timestamps:
        angle = 0.2*t
        radius = 50 + 1.0*t  # increasing spiral radius
        pos = start_D2 + np.array([radius*np.cos(angle), radius*np.sin(angle), 1.5*t])
        noise = np.random.normal(0, 1.5, 3)
        pos = pos + noise
        rows.append({
            'time': t,
            'id': 'D2',
            'type': 'DJI Mavic',
            'x': pos[0],
            'y': pos[1],
            'z': pos[2]
        })

    # Drone D3: Surveillance (Recon) with steady drift and altitude oscillation.
    start_D3 = np.array([-300, 600, 150])
    for t in timestamps:
        base = start_D3 + np.array([12*t, -14*t, 1.2*t])
        oscillation = np.array([10*np.sin(0.15*t), 10*np.cos(0.15*t), 4*np.sin(0.2*t)])
        noise = np.random.normal(0, 1, 3)
        pos = base + oscillation + noise
        rows.append({
            'time': t,
            'id': 'D3',
            'type': 'Recon',
            'x': pos[0],
            'y': pos[1],
            'z': pos[2]
        })
    
    df = pd.DataFrame(rows).sort_values(["id", "time"]).reset_index(drop=True)
    df[['velocity_x', 'velocity_y', 'velocity_z']] = df.groupby('id')[['x','y','z']].diff().fillna(0)
    return df

df = make_dummy_data()

#-----------------------------------------------------------
# 3. Friendly Unit Positions (Local Coords in meters)
#-----------------------------------------------------------
friendly_units = {
    "Alpha (1st Battalion HQ)": (-50, -50, 0),
    "Bravo (FOB – Forward Operating Base)": (250, 150, 0),
    "Charlie (Observation Post)": (-200, -150, 0)
}

#-----------------------------------------------------------
# 4. Compute Projected Impact Zone for a High-threat Drone (Shahed)
#-----------------------------------------------------------
def compute_impact_zone(x, y, z, vx, vy, vz, seconds=5):
    proj_x = x + vx * seconds
    proj_y = y + vy * seconds
    proj_z = z + vz * seconds
    displacement = math.sqrt((vx*seconds)**2 + (vy*seconds)**2 + (vz*seconds)**2)
    impact_radius = 0.1 * displacement + 20
    return proj_x, proj_y, proj_z, impact_radius

#-----------------------------------------------------------
# 5. Plotting Functions: 2D Map & 3D View
#-----------------------------------------------------------
def get_recent_traj(drone_id, current_time, window=5):
    """Return the trajectory (last few points) for a given drone id."""
    traj = df[(df['id'] == drone_id) & (df.time <= current_time)].tail(window)
    return traj

def build_map_figure(t, ref_location):
    ref_lat = ref_location["lat"]
    ref_lon = ref_location["lon"]
    frame = df[df.time == t]

    # Drone markers, impact zones, and trajectory tails holders.
    drone_lats, drone_lons, drone_texts, drone_colors = [], [], [], []
    traj_xs, traj_ys, traj_texts, traj_colors = {}, {}, {}, {}

    impact_circles = []  

    # Process each row of current frame.
    for _, row in frame.iterrows():
        lat, lon = to_geo(row.x, row.y, ref_lat, ref_lon)
        drone_lats.append(lat)
        drone_lons.append(lon)
        vel_mag = math.sqrt(row.velocity_x**2 + row.velocity_y**2 + row.velocity_z**2)
        drone_texts.append(f"{row['id']} ({row['type']})<br>Alt: {row.z:.1f}m<br>Speed: {vel_mag:.1f} m/s")
        
        # Color coding.
        if row['type'] == 'Shahed':
            color = "red"
            # Compute impact zone.
            proj_x, proj_y, proj_z, impact_radius = compute_impact_zone(row.x, row.y, row.z, row.velocity_x, row.velocity_y, row.velocity_z, seconds=5)
            proj_lat, proj_lon = to_geo(proj_x, proj_y, ref_lat, ref_lon)
            impact_radius_deg = impact_radius / 111000
            impact_circles.append({
                'center_lat': proj_lat,
                'center_lon': proj_lon,
                'radius_deg': impact_radius_deg,
                'text': f"{row['id']} Impact Zone<br>Radius: {impact_radius:.1f} m"
            })
        elif row['type'] == 'DJI Mavic':
            color = "blue"
        else:
            color = "green"
        drone_colors.append(color)
        
        # Trajectory tail: save last window points for this drone.
        traj = get_recent_traj(row.id, t)
        traj_lats = []
        traj_lons = []
        for _, srow in traj.iterrows():
            glat, glon = to_geo(srow.x, srow.y, ref_lat, ref_lon)
            traj_lats.append(glat)
            traj_lons.append(glon)
        traj_xs[row.id] = traj_lats
        traj_ys[row.id] = traj_lons
        traj_texts[row.id] = row['id']
        traj_colors[row.id] = color

    # Friendly unit positions.
    friendly_lats, friendly_lons, friendly_texts = [], [], []
    for name, (fx, fy, _) in friendly_units.items():
        glat, glon = to_geo(fx, fy, ref_lat, ref_lon)
        friendly_lats.append(glat)
        friendly_lons.append(glon)
        friendly_texts.append(name)
    
    # Build the map figure.
    fig = go.Figure()

    # Drone current positions.
    fig.add_trace(go.Scattermapbox(
        lat=drone_lats,
        lon=drone_lons,
        mode='markers',
        marker=go.scattermapbox.Marker(size=12, color=drone_colors),
        text=drone_texts,
        hoverinfo='text',
        name='Drones'
    ))
    
    # Drone trajectory tails.
    for drone_id, lats in traj_xs.items():
        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=traj_ys[drone_id],
            mode='lines+markers',
            line=dict(color=traj_colors[drone_id], width=2),
            marker=dict(size=6),
            name=f"{drone_id} Trajectory",
            hoverinfo='none'
        ))
    
    # Friendly units.
    fig.add_trace(go.Scattermapbox(
        lat=friendly_lats,
        lon=friendly_lons,
        mode='markers+text',
        marker=go.scattermapbox.Marker(size=14, color="black", symbol="star"),
        text=friendly_texts,
        textposition="top right",
        name="Friendly Units",
        hoverinfo="text"
    ))
    
    # Impact zones for high-threat drones.
    for circle in impact_circles:
        circle_lats = []
        circle_lons = []
        N = 50
        for theta in np.linspace(0, 2*np.pi, N):
            circle_lats.append(circle['center_lat'] + circle['radius_deg'] * np.cos(theta))
            circle_lons.append(circle['center_lon'] + circle['radius_deg'] * np.sin(theta))
        fig.add_trace(go.Scattermapbox(
            lat=circle_lats,
            lon=circle_lons,
            mode='lines',
            line=dict(color="red", width=2),
            name="Impact Zone",
            hoverinfo='none'
        ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=go.layout.mapbox.Center(lat=ref_lat, lon=ref_lon),
            zoom=10
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        title=f"2D Drone Tracker – Time = {t} s"
    )
    return fig, frame

def build_3d_figure(t):
    # Use a 3D scatter (axes are local cartesian x,y,z) with trajectories.
    frame = df[df.time == t]
    
    fig = go.Figure()

    # Process each drone.
    drone_ids = df['id'].unique()
    for did in drone_ids:
        traj = df[(df.id == did) & (df.time <= t)]
        if traj.empty:
            continue
        # Choose color.
        color = "red" if traj.iloc[-1]['type'] == "Shahed" else ("blue" if traj.iloc[-1]['type'] == "DJI Mavic" else "green")
        fig.add_trace(go.Scatter3d(
            x=traj['x'],
            y=traj['y'],
            z=traj['z'],
            mode='lines+markers',
            marker=dict(size=4, color=color),
            line=dict(color=color, width=3),
            name=f"{did} Trajectory",
            hoverinfo='text',
            text=[f"{did} t={t_val}" for t_val in traj['time']]
        ))
        # For high-threat drones, add projected impact sphere.
        last = traj.iloc[-1]
        if last['type'] == "Shahed":
            proj_x, proj_y, proj_z, impact_radius = compute_impact_zone(last.x, last.y, last.z, last.velocity_x, last.velocity_y, last.velocity_z, seconds=5)
            # Create sphere mesh (approximation).
            u = np.linspace(0, 2*np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            xs = proj_x + impact_radius * np.outer(np.cos(u), np.sin(v))
            ys = proj_y + impact_radius * np.outer(np.sin(u), np.sin(v))
            zs = proj_z + impact_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            # Use wireframe as a surface.
            fig.add_trace(go.Surface(
                x=xs,
                y=ys,
                z=zs,
                colorscale=[[0, 'red'], [1, 'red']],
                opacity=0.3,
                showscale=False,
                name=f"{last['id']} Impact Zone",
                hoverinfo='skip'
            ))
    
    # Add friendly units as 3D markers.
    for name, (fx, fy, fz) in friendly_units.items():
        fig.add_trace(go.Scatter3d(
            x=[fx],
            y=[fy],
            z=[fz],
            mode='markers+text',
            marker=dict(size=8, color="black", symbol="diamond"),
            text=[name],
            textposition="top center",
            name="Friendly Unit"
        ))
    
    fig.update_layout(
        title=f"3D Drone Tracker – Time = {t} s",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Altitude (m)",
            xaxis=dict(range=[-700,700]),
            yaxis=dict(range=[-700,700]),
            zaxis=dict(range=[0,400])
        ),
        margin={"r":0,"t":30,"l":0,"b":0}
    )
    return fig, frame

#-----------------------------------------------------------
# 6. Streamlit Interface with Tabs for 2D and 3D Views
#-----------------------------------------------------------
st.set_page_config(page_title="Drone Intelligence Dashboard", layout="wide")
st.title("Drone Intelligence Dashboard – Multi-View")

st.markdown("""
This dashboard monitors enemy drone activity in both geographic (2D) and local cartesian (3D) views.  
Use the controls in the sidebar to select the region, time, and simulate in real time.
""")

# Sidebar Controls:
st.sidebar.header("Simulation Controls")
selected_region = st.sidebar.selectbox("Select Base Region", list(reference_locations.keys()))
ref_location = reference_locations[selected_region]
t_slider = st.sidebar.slider("Select Time (s)", 0, int(df.time.max()), 0, step=1)
play = st.sidebar.button("Play Live Simulation")
update_interval = st.sidebar.number_input("Simulation Speed (seconds per frame)", min_value=0.1, max_value=5.0, value=0.75, step=0.1)

# Create two tabs: one for 2D Map and one for 3D view.
tab2d, tab3d = st.tabs(["2D Map View", "3D View"])

# Placeholders for each tab to update without clearing the entire layout.
with tab2d:
    map_placeholder = st.empty()
with tab3d:
    fig3d_placeholder = st.empty()

# Function to generate a textual summary for current frame.
def generate_summary(frame):
    lines = []
    drone_counts = frame['type'].value_counts().to_dict()
    lines.append("**Drone Counts:**")
    for dtype, count in drone_counts.items():
        lines.append(f"- {dtype}: {count}")
    lines.append("**Nearest Drone (by friendly unit):**")
    for name, pos in friendly_units.items():
        min_dist = float("inf")
        closest = None
        for _, row in frame.iterrows():
            d = math.sqrt((pos[0]-row.x)**2 + (pos[1]-row.y)**2 + (pos[2]-row.z)**2)
            if d < min_dist:
                min_dist = d
                closest = f"{row['id']} ({row['type']})"
        lines.append(f"- {name}: {closest} ({min_dist:.1f} m)")
    return "\n".join(lines)

details_placeholder = st.empty()

#-----------------------------------------------------------
# 7. Simulation Update Loop
#-----------------------------------------------------------
if play:
    max_time = int(df.time.max())
    # Run simulation from the slider value to max_time.
    for t in range(t_slider, max_time + 1):
        fig2d, frame2d = build_map_figure(t, ref_location)
        fig3d, frame3d = build_3d_figure(t)
        map_placeholder.plotly_chart(fig2d, use_container_width=True)
        fig3d_placeholder.plotly_chart(fig3d, use_container_width=True)
        details_placeholder.markdown(generate_summary(frame2d))
        time.sleep(update_interval)
    st.session_state["t_slider"] = max_time
else:
    fig2d, frame2d = build_map_figure(t_slider, ref_location)
    fig3d, frame3d = build_3d_figure(t_slider)
    map_placeholder.plotly_chart(fig2d, use_container_width=True)
    fig3d_placeholder.plotly_chart(fig3d, use_container_width=True)
    details_placeholder.markdown(generate_summary(frame2d))
