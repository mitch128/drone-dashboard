import plotly.graph_objects as go

fig = go.Figure(go.Scattermapbox(
        lat=[ORIGIN_LAT],
        lon=[ORIGIN_LON],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=9
        ),
        text=["Kursk, Russia"],
    ))

fig.update_layout(
    mapbox_style="open-street-map",
    mapbox=dict(
        center=dict(lat=ORIGIN_LAT, lon=ORIGIN_LON),
        zoom=10,
    ),
    margin={"r":0,"t":0,"l":0,"b":0}
)

st.plotly_chart(fig, use_container_width=True)
