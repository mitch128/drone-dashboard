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
