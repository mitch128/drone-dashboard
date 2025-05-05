import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time, math

#######################################
# 0. Global Settings & Conversion from Local XY (in meters) to Geographic Coordinates
#######################################
# Kursk, Russia baseline; adjust as needed.
ORIGIN_LAT = 51.7300
ORIGIN_LON = 36.1900

def meters_to_latlon(x, y, origin_lat=ORIGIN_LAT, origin_lon=ORIGIN_LON):
    # 1 degree latitude is ~111111 meters
    dlat = y / 111111
    # 1 degree longitude is ~111111 * cos(latitude) meters
    dlon = x / (111111 * math.cos(math.radians(origin_lat)))
    return origin_lat + dlat, origin_lon + dlon

#######################################
# 1. Generate More Realistic Dummy Drone Data in 3D
#######################################
def make_dummy_data():
    np.random.seed(42)  # reproducibility
    timestamps = np.arange(0, 31)  # Simulation for 30 seconds
    rows = []
    
    # Drone D1 (High-threat Shahed) – curved trajectory with descent.
    start_D1 = np.array([-600, -500, 200])
    base_velocity_D1 = np.array([30, 35, -4])
    for t in timestamps:
        deviation = np.array([15 * np.sin(0.1*t), 15 * np.cos(0.1*t), 5 * np.sin(0.05*t)])
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
        
    # Drone D2 (Recon DJI Mavic) – looping spiral with gradual outward drift.
    start_D2 = np.array([200, 400, 100])
    for t in timestamps:
        angle = 0.2 * t
        radius = 50 + 0.5 * t   # increasing radius
        pos = start_D2 + np.array([radius * np.cos(angle), radius * np.sin(angle), 2*t])
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
    
    # Drone D3 (Surveillance Recon) – straight drift with sinusoidal oscillations.
    start_D3 = np.array([-300, 600, 150])
    for t in timestamps:
        base = start_D3 + np.array([10*t, -12*t, t])
        oscillation = np.array([8*np.sin(0.15*t), 8*np.cos(0.15*t), 3*np.sin(0.2*t)])
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
    
    df = pd.DataFrame(rows)
    
    # Calculate approximate velocities using backward differences.
    df[['velocity_x', 'velocity_y', 'velocity_z']] = df.groupby('id')[['x','y','z']].diff().fillna(0)
    
    # Compute geographic coordinates for mapping (lat, lon)
    df['lat'], df['lon'] = zip(*df.apply(lambda row: meters_to_latlon(row.x, row.y), axis=1))
    
    return df

df = make_dummy_data()

#######################################
# 2. Define Friendly Positions (Frontline Units) in 3D with Geographic Conversion
#######################################
# Local friendly positions (on the ground, z=0) relative to (0,0)
infantry_local = {
    "Alpha (1st Battalion HQ)": (0, 0, 0),
    "Bravo (FOB – Forward Operating Base)": (250, 150, 0),
    "Charlie (Observation Post)": (-200, -150, 0)
}

# Convert friendly positions to geographic coordinates.
infantry_positions = {}
for name, (ix, iy, iz) in infantry_local.items():
    lat, lon = meters_to_latlon(ix, iy)
    infantry_positions[name] = {'local': (ix, iy, iz), 'lat': lat, 'lon': lon}

#######################################
# 3. Utility Functions
#######################################
def distance_3d(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def compute_impact_zone(x, y, z, vx, vy, vz, seconds=5):
    proj_x = x + vx * seconds
    proj_y = y + vy * seconds
    proj_z = z + vz * seconds
    displacement = math.sqrt((vx*seconds)**2 + (vy*seconds)**2 + (vz*seconds)**2)
    impact_radius = 0.1 * displacement + 20
    return proj_x, proj_y, proj_z, impact_radius

#######################################
# 4. Plotting Functions
#######################################

# 4a. 2D Map Plot using Plotly Mapbox
def plot_map_frame(t):
    frame = df[df.time == t]
    fig = go.Figure()

    # Plot friendly units on the map.
    for name, pos in infantry_positions.items():
        fig.add_trace(go.Scattermapbox(
            lat=[pos['lat']],
            lon=[pos['lon']],
            mode="markers+text",
            marker=dict(size=12, symbol="harbor", color="black"),
            text=[name],
            textposition="top right",
            name=name
        ))
    
    # Plot each drone’s trajectory.
    for drone_id in df['id'].unique():
        drone_data = df[(df.id == drone_id) & (df.time <= t)].sort_values(by='time')
        drone_type = drone_data.iloc[0]['type']
        traj_color = {"Shahed": "red", "DJI Mavic": "blue", "Recon": "green"}.get(drone_type, "gray")
        
        # If more than one point exists, plot as line+markers, else just markers.
        if len(drone_data) > 1:
            fig.add_trace(go.Scattermapbox(
                lat=drone_data['lat'],
                lon=drone_data['lon'],
                mode="lines+markers",
                marker=dict(size=8, color=traj_color),
                line=dict(width=2, color=traj_color),
                name=f"{drone_id} ({drone_type})"
            ))
        else:
            fig.add_trace(go.Scattermapbox(
                lat=drone_data['lat'],
                lon=drone_data['lon'],
                mode="markers",
                marker=dict(size=10, color=traj_color),
                name=f"{drone_id} ({drone_type})"
            ))
        
        # For high-threat drones, add projected impact zone.
        if drone_type == "Shahed" and not drone_data.empty:
            current = drone_data.iloc[-1]
            vx = current.velocity_x
            vy = current.velocity_y
            vz = current.velocity_z
            proj_x, proj_y, proj_z, impact_radius = compute_impact_zone(current.x, current.y, current.z, vx, vy, vz, seconds=5)
            proj_lat, proj_lon = meters_to_latlon(proj_x, proj_y)
            
            # Add an arrow-like marker (using text annotation for simplicity)
            fig.add_trace(go.Scattermapbox(
                lat=[proj_lat],
                lon=[proj_lon],
                mode="markers+text",
                marker=dict(size=12, color="red", symbol="cross"),
                text=["Impact Zone"],
                textposition="top right",
                name=f"{drone_id} Impact Zone"
            ))
            
            # Create circle for impact zone.
            circle_lats, circle_lons = [], []
            for deg in np.linspace(0, 360, 40):
                angle_rad = math.radians(deg)
                dx = impact_radius * math.cos(angle_rad)
                dy = impact_radius * math.sin(angle_rad)
                circ_lat, circ_lon = meters_to_latlon(proj_x + dx, proj_y + dy)
                circle_lats.append(circ_lat)
                circle_lons.append(circ_lon)
            fig.add_trace(go.Scattermapbox(
                lat=circle_lats,
                lon=circle_lons,
                mode="lines",
                line=dict(color="red", dash="dot"),
                name=f"{drone_id} Impact Radius"
            ))
    
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=ORIGIN_LAT, lon=ORIGIN_LON),
            zoom=10,
        ),
        margin={"r":0, "t":0, "l":0, "b":0},
        height=600
    )
    return fig, frame

# 4b. 3D Trajectory Plot using Plotly 3D
def plot_3d_frame(t):
    frame = df[df.time == t]
    fig = go.Figure()
    
    # Plot friendly positions (ground units)
    for name, pos in infantry_local.items():
        fig.add_trace(go.Scatter3d(
            x=[pos[0]],
            y=[pos[1]],
            z=[pos[2]],
            mode='markers+text',
            text=[name],
            marker=dict(size=8, symbol="diamond", color='black'),
            name=name
        ))
    
    # Plot drone trajectories.
    for drone_id in df['id'].unique():
        drone_data = df[(df.id == drone_id) & (df.time <= t)].sort_values(by='time')
        drone_type = drone_data.iloc[0]['type']
        traj_color = {"Shahed": "red", "DJI Mavic": "blue", "Recon": "green"}.get(drone_type, "gray")
        
        fig.add_trace(go.Scatter3d(
            x=drone_data['x'],
            y=drone_data['y'],
            z=drone_data['z'],
            mode='lines+markers',
            marker=dict(size=5, color=traj_color),
            line=dict(width=3, color=traj_color),
            name=f"{drone_id} ({drone_type})"
        ))
        
        # For high-threat drones, add a projected impact vector.
        if drone_type == "Shahed" and not drone_data.empty:
            current = drone_data.iloc[-1]
            vx = current.velocity_x
            vy = current.velocity_y
            vz = current.velocity_z
            proj_x, proj_y, proj_z, impact_radius = compute_impact_zone(current.x, current.y, current.z, vx, vy, vz, seconds=5)
            fig.add_trace(go.Cone(
                x=[current.x],
                y=[current.y],
                z=[current.z],
                u=[proj_x - current.x],
                v=[proj_y - current.y],
                w=[proj_z - current.z],
                colorscale=[[0, 'red'], [1, 'red']],
                sizemode="absolute",
                sizeref=impact_radius/5,
                showscale=False,
                name=f"{drone_id} projected"
            ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Altitude (m)",
            xaxis=dict(range=[-700,700]),
            yaxis=dict(range=[-700,700]),
            zaxis=dict(range=[0,400])
        ),
        margin=dict(r=10,l=10,b=10,t=30),
        height=600,
        title=f"3D Drone Trajectories – Time = {t}s"
    )
    return fig, frame

#######################################
# 5. Streamlit UI – Command Center Dashboard
#######################################
st.set_page_config(page_title="Drone Intelligence Dashboard", layout="wide")
st.title("Drone Intelligence Dashboard")
st.markdown("""
This tool monitors enemy drone activity over a map area and in a 3D view relative to frontline units.  
Use the controls below to play or step through the simulation.
""")

# Sidebar Controls
st.sidebar.header("Command Center Controls")
view_mode = st.sidebar.radio("Display Mode", options=["2D Map", "3D View"], index=0)
t_slider = st.sidebar.slider("Select Time (s)", 0, int(df.time.max()), 0, 1)
play = st.sidebar.button("Play Live Simulation")
update_interval = st.sidebar.number_input("Simulation Speed (seconds per frame)", min_value=0.1, max_value=5.0, value=0.75, step=0.1)

# Layout for visualization and summary.
col_vis, col_summary = st.columns([2, 1])
with col_vis:
    vis_placeholder = st.empty()
with col_summary:
    st.subheader("Battlefield Summary")
    summary_placeholder = st.empty()
    st.subheader("Event Log")
    event_placeholder = st.empty()

def generate_summary(frame):
    summary_lines = []
    drone_counts = frame.groupby("type")["id"].nunique().to_dict()
    summary_lines.append("**Current Drone Counts:**")
    for dtype, cnt in drone_counts.items():
        summary_lines.append(f"- {dtype}: {cnt}")
    summary_lines.append("\n**Nearest Drone per Unit:**")
    for name, pos in infantry_local.items():
        min_dist = float("inf")
        closest_drone = None
        for _, row in frame.iterrows():
            d = distance_3d(pos, (row.x, row.y, row.z))
            if d < min_dist:
                min_dist = d
                closest_drone = f"{row['type']} ({row['id']})"
        if closest_drone:
            summary_lines.append(f"- {name}: {closest_drone} at {min_dist:.1f} m")
        else:
            summary_lines.append(f"- {name}: No drone detected")
    return "\n".join(summary_lines)

#######################################
# 6. Real-time Simulation / Slider Update
#######################################
if play:
    max_time = int(df.time.max())
    for t in range(t_slider, max_time+1):
        if view_mode == "2D Map":
            fig, frame = plot_map_frame(t)
        else:
            fig, frame = plot_3d_frame(t)
        vis_placeholder.plotly_chart(fig, use_container_width=True)
        summary_placeholder.markdown(generate_summary(frame))
        
        events = []
        for idx, row in frame.iterrows():
            if row['type'] == "Shahed":
                vx = row.velocity_x
                vy = row.velocity_y
                vz = row.velocity_z
                proj_x, proj_y, proj_z, radius = compute_impact_zone(row.x, row.y, row.z, vx, vy, vz, seconds=5)
                events.append(f"ALERT: {row['id']} ({row['type']}) projected impact at ({proj_x:.1f}, {proj_y:.1f}, {proj_z:.1f}) with radius {radius:.1f}m")
        if events:
            event_placeholder.markdown("\n".join([f"- {e}" for e in events]))
        else:
            event_placeholder.markdown("No new alerts at this time.")
        time.sleep(update_interval)
    st.session_state["time_slider"] = max_time
else:
    if view_mode == "2D Map":
        fig, frame = plot_map_frame(t_slider)
    else:
        fig, frame = plot_3d_frame(t_slider)
    vis_placeholder.plotly_chart(fig, use_container_width=True)
    summary_placeholder.markdown(generate_summary(frame))
    
    events = []
    for idx, row in frame.iterrows():
        if row['type'] == "Shahed":
            vx = row.velocity_x
            vy = row.velocity_y
            vz = row.velocity_z
            proj_x, proj_y, proj_z, radius = compute_impact_zone(row.x, row.y, row.z, vx, vy, vz, seconds=5)
            events.append(f"ALERT: {row['id']} ({row['type']}) projected impact at ({proj_x:.1f}, {proj_y:.1f}, {proj_z:.1f}) with radius {radius:.1f}m")
    if events:
        event_placeholder.markdown("\n".join([f"- {e}" for e in events]))
    else:
        event_placeholder.markdown("No critical events at this time.")
