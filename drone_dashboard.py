import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import math

#-----------------------------------------------------------
# 1. Define Simulation Parameters & Geographic Helpers
#-----------------------------------------------------------
# Reference locations (lat, lon) for Kursk, Russia and a few alternatives.
reference_locations = {
    "Kursk, Russia": {"lat": 51.7300, "lon": 36.1900},
    "Odessa, Ukraine": {"lat": 46.4825, "lon": 30.7233},
    "Kiev, Ukraine": {"lat": 50.4501, "lon": 30.5234},
}

# For the simulation we will convert the local (x,y) coordinates into small offsets (in degrees) from the chosen ref.
def to_geo(x, y, ref_lat, ref_lon):
    # Assume 1 degree latitude ~ 111 km, 1 degree longitude ~ (cos(lat)*111 km).
    delta_lat = x / 111000  # x offset in meters -> degrees latitude
    delta_lon = y / (111000 * np.cos(np.deg2rad(ref_lat)))
    return ref_lat + delta_lat, ref_lon + delta_lon

#-----------------------------------------------------------
# 2. Generate More Realistic Dummy Drone Data (30-second simulation)
#-----------------------------------------------------------
def make_dummy_data():
    # We simulate 3 drones with id and type over 30 sec in a local coordinate system (in meters)
    np.random.seed(42)  # reproducibility
    timestamps = np.arange(0, 31)  # 0 to 30 seconds
    rows = []
    
    # Drone D1: High-threat (Shahed) with a curved, descending trajectory.
    start_D1 = np.array([-600, -500, 200])   # (x,y,z) starting in meters relative to ref point.
    base_velocity_D1 = np.array([30, 35, -4])  # constant components per sec
    for t in timestamps:
        # Introduce slight curvature using sinusoids plus minor noise.
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
    
    # Drone D2: Recon drone (DJI Mavic) executing a looping spiral with steady altitude drift.
    start_D2 = np.array([200, 400, 100])
    for t in timestamps:
        angle = 0.2*t
        radius = 50 + 1.0*t  # gradual increase in spiral radius
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

    # Drone D3: Surveillance drone (Recon) with steady drift and sinusoidal oscillation in altitude.
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
    
    # Compute velocity components using backward differences.
    df[['velocity_x', 'velocity_y', 'velocity_z']] = df.groupby('id')[['x','y','z']].diff().fillna(0)
    return df

df = make_dummy_data()

#-----------------------------------------------------------
# 3. Define Friendly Unit Positions (in our local coordinate system, meters)
#-----------------------------------------------------------
# Assume friendly units are on the ground (z=0). Positions are relative to the simulation reference point.
friendly_units = {
    "Alpha (1st Battalion HQ)": (-50, -50, 0),
    "Bravo (FOB – Forward Operating Base)": (250, 150, 0),
    "Charlie (Observation Post)": (-200, -150, 0)
}

#-----------------------------------------------------------
# 4. Compute Projected Impact Zone for High-threat Drones
#-----------------------------------------------------------
def compute_impact_zone(x, y, z, vx, vy, vz, seconds=5):
    # Extrapolate future position
    proj_x = x + vx * seconds
    proj_y = y + vy * seconds
    proj_z = z + vz * seconds
    # Define impact radius (meters) based on displacement and uncertainty
    displacement = math.sqrt((vx*seconds)**2 + (vy*seconds)**2 + (vz*seconds)**2)
    impact_radius = 0.1 * displacement + 20  # add base uncertainty
    return proj_x, proj_y, proj_z, impact_radius

#-----------------------------------------------------------
# 5. Produce a Map with Overlays using Plotly
#-----------------------------------------------------------
def plot_map_frame(t, ref_location):
    # Filter data for time t
    frame = df[df.time == t]
    
    # Get reference lat/lon from chosen location
    ref_lat = ref_location["lat"]
    ref_lon = ref_location["lon"]

    # Convert drone positions to geographic coordinates
    lats = []
    lons = []
    texts = []
    markers = []
    impact_circles = []  # only for Shahed drones
    for _, row in frame.iterrows():
        lat, lon = to_geo(row.x, row.y, ref_lat, ref_lon)
        lats.append(lat)
        lons.append(lon)
        vel_mag = math.sqrt(row.velocity_x**2 + row.velocity_y**2 + row.velocity_z**2)
        text = f"{row['id']} ({row['type']})<br>Alt: {row.z:.1f} m<br>Speed: {vel_mag:.1f} m/s"
        texts.append(text)
        # Define marker color based on type.
        if row['type'] == 'Shahed':
            markers.append("red")
            # Compute and store the impact zone for this high-threat drone.
            proj_x, proj_y, proj_z, impact_radius = compute_impact_zone(row.x, row.y, row.z, row.velocity_x, row.velocity_y, row.velocity_z, seconds=5)
            proj_lat, proj_lon = to_geo(proj_x, proj_y, ref_lat, ref_lon)
            # Impact circle details: center and radius (converted to degrees approximation).
            # 1 degree lat ~111 km.
            impact_radius_deg = impact_radius / 111000
            impact_circles.append({
                'lat': proj_lat,
                'lon': proj_lon,
                'radius_deg': impact_radius_deg,
                'drone_text': f"{row['id']} Impact Zone<br>Radius: {impact_radius:.1f} m"
            })
        elif row['type'] == 'DJI Mavic':
            markers.append("blue")
        else:
            markers.append("green")
            
    # Convert friendly unit positions to geo coordinates.
    friendly_lats = []
    friendly_lons = []
    friendly_texts = []
    for name, (fx, fy, fz) in friendly_units.items():
        lat, lon = to_geo(fx, fy, ref_lat, ref_lon)
        friendly_lats.append(lat)
        friendly_lons.append(lon)
        friendly_texts.append(name)
    
    # Create Plotly figure using scatter_mapbox
    fig = go.Figure()

    # Add drone positions (3D properties displayed in tooltip)
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='markers+text',
        marker=go.scattermapbox.Marker(
            size=12,
            color=markers,
            opacity=0.9
        ),
        text=[d for d in texts],
        hoverinfo='text',
        name="Drones"
    ))
    
    # Add friendly units
    fig.add_trace(go.Scattermapbox(
        lat=friendly_lats,
        lon=friendly_lons,
        mode='markers+text',
        marker=go.scattermapbox.Marker(
            size=14,
            color="black",
            symbol="star"
        ),
        text=friendly_texts,
        hoverinfo='text',
        name="Friendly Units"
    ))
    
    # Add impact zone circles for high-threat drones
    for circle in impact_circles:
        # Build a circle polygon (approximate with 50 points).
        circle_lats = []
        circle_lons = []
        for theta in np.linspace(0, 2*np.pi, 50):
            circle_lats.append(circle['lat'] + circle['radius_deg'] * np.cos(theta))
            circle_lons.append(circle['lon'] + circle['radius_deg'] * np.sin(theta))
        # Close the loop
        circle_lats.append(circle_lats[0])
        circle_lons.append(circle_lons[0])
        
        fig.add_trace(go.Scattermapbox(
            lat=circle_lats,
            lon=circle_lons,
            mode='lines',
            line=go.scattermapbox.Line(color="red", width=2),
            hoverinfo='none',
            name="Projected Impact Zone"
        ))
    
    # Set layout with automatically chosen zoom and style. Using "open-street-map" so no token is needed.
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=go.layout.mapbox.Center(lat=ref_lat, lon=ref_lon),
            zoom=10
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        title=f"Drone Tracker – Time = {t} s"
    )
    
    return fig, frame

#-----------------------------------------------------------
# 6. Build Streamlit Interface
#-----------------------------------------------------------
st.set_page_config(page_title="Drone Intelligence Dashboard", layout="wide")
st.title("Drone Intelligence Dashboard – Interactive Map View")
st.markdown("""
This dashboard monitors enemy drone activity relative to deployed friendly units using real geographic overlays.  
Select the area of interest and time to view the current drone positions and their projected impact zones (for high-threat drones).
""")

# Sidebar Controls:
st.sidebar.header("Simulation Controls")

# Choose the base geographic region.
selected_region = st.sidebar.selectbox("Select Base Region", list(reference_locations.keys()))
ref_location = reference_locations[selected_region]

# Slider for simulation time
t_slider = st.sidebar.slider("Select Time (s)", 0, int(df.time.max()), 0, step=1)

play = st.sidebar.button("Play Live Simulation")
update_interval = st.sidebar.number_input("Simulation Speed (seconds per frame)", min_value=0.1, max_value=5.0, 
                                            value=0.75, step=0.1)

# Layout: Left for Map, Right for details and logs.
col_map, col_summary = st.columns([2, 1])
with col_map:
    map_placeholder = st.empty()
with col_summary:
    st.subheader("Battlefield Summary")
    summary_placeholder = st.empty()
    st.subheader("Event Log")
    event_placeholder = st.empty()

# Utility: Euclidean Distance between a friendly unit and a drone (local coordinates)
def distance_3d(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

# Build a textual summary
def generate_summary(frame):
    summary_lines = []
    # Count drones by type:
    drone_counts = frame['type'].value_counts().to_dict()
    summary_lines.append("**Current Drone Counts:**")
    for dtype, count in drone_counts.items():
        summary_lines.append(f"- {dtype}: {count}")
        
    # Nearest drone per friendly unit (using local coords)
    summary_lines.append("\n**Nearest Drone per Friendly Unit:**")
    for name, pos in friendly_units.items():
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

#-----------------------------------------------------------
# 7. Real-Time Simulation / Frame Update Logic
#-----------------------------------------------------------
if play:
    max_time = int(df.time.max())
    for t in range(t_slider, max_time + 1):
        fig, frame = plot_map_frame(t, ref_location)
        map_placeholder.plotly_chart(fig, use_container_width=True)
        summary_placeholder.markdown(generate_summary(frame))
        
        # Log events for high-threat drones (Shahed)
        events = []
        for _, row in frame.iterrows():
            if row['type'] == 'Shahed':
                vx, vy, vz = row.velocity_x, row.velocity_y, row.velocity_z
                proj_x, proj_y, proj_z, impact_radius = compute_impact_zone(row.x, row.y, row.z, vx, vy, vz, seconds=5)
                events.append(f"ALERT: {row['id']} [{row['type']}] projected impact zone at approx. ({proj_x:.1f}, {proj_y:.1f}, {proj_z:.1f}) m, radius {impact_radius:.1f} m")
        if events:
            event_placeholder.markdown("\n".join([f"- {e}" for e in events]))
        else:
            event_placeholder.markdown("No new alerts at this time.")
        time.sleep(update_interval)
    st.session_state["time_slider"] = max_time
else:
    fig, frame = plot_map_frame(t_slider, ref_location)
    map_placeholder.plotly_chart(fig, use_container_width=True)
    summary_placeholder.markdown(generate_summary(frame))
    events = []
    for _, row in frame.iterrows():
        if row['type'] == 'Shahed':
            vx, vy, vz = row.velocity_x, row.velocity_y, row.velocity_z
            proj_x, proj_y, proj_z, impact_radius = compute_impact_zone(row.x, row.y, row.z, vx, vy, vz, seconds=5)
            events.append(f"ALERT: {row['id']} [{row['type']}] projected impact zone at approx. ({proj_x:.1f}, {proj_y:.1f}, {proj_z:.1f}) m, radius {impact_radius:.1f} m")
    if events:
        event_placeholder.markdown("\n".join([f"- {e}" for e in events]))
    else:
        event_placeholder.markdown("No critical events at this time.")
