import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import plotly.graph_objects as go

# === 1. Load dataset ===
df = pd.read_csv("processed_data/suburb_yearly_bybednum_sydney_enriched.csv")
df.columns = df.columns.str.strip()
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["median"] = pd.to_numeric(df["median"], errors="coerce")
df = df.dropna(subset=["year", "median"])
df = df[(df["year"] >= 2000) & (df["year"] <= 2022)]
years_sorted = sorted(df["year"].unique())
df["year"] = pd.Categorical(df["year"], categories=years_sorted, ordered=True)

# === 2. Prepare figure and frames ===
fig = go.Figure()
frames = []
year_to_ymax = {}

for year in years_sorted:
    subset = df[df["year"] == year]["median"].dropna()
    if len(subset) < 2:
        continue

    # Histogram
    hist_y, hist_x = np.histogram(subset, bins=50)
    hist_x_center = 0.5 * (hist_x[:-1] + hist_x[1:])

    # Density
    kde = gaussian_kde(subset)
    x_range = np.linspace(subset.min(), subset.max(), 200)
    y_density = kde(x_range)
    y_density_scaled = y_density * hist_y.max() / y_density.max()

    # Compute padded y_max
    y_max = max(hist_y.max(), y_density_scaled.max()) * 1.1
    year_to_ymax[year] = y_max

    # Frame with per-frame layout (for smooth yaxis interpolation)
    frame = go.Frame(
    data=[
        go.Bar(x=hist_x_center, y=hist_y, marker_color="#69b3ff"),
        go.Scatter(x=x_range, y=y_density_scaled, mode="lines",
                   line=dict(color="darkblue", width=3))
    ],
    name=str(year),
    layout=go.Layout(
        yaxis=dict(range=[0, y_max], autorange=False)  # <- jump to new range
    )
    )
    frames.append(frame)

# === 3. Add first frame ===
first_year = int(frames[0].name)
subset = df[df["year"] == first_year]["median"].dropna()
hist_y, hist_x = np.histogram(subset, bins=50)
hist_x_center = 0.5 * (hist_x[:-1] + hist_x[1:])
kde = gaussian_kde(subset)
x_range = np.linspace(subset.min(), subset.max(), 200)
y_density = kde(x_range)
y_density_scaled = y_density * hist_y.max() / y_density.max()

fig.add_trace(go.Bar(x=hist_x_center, y=hist_y, marker_color="#69b3ff"))
fig.add_trace(go.Scatter(x=x_range, y=y_density_scaled, mode="lines",
                         line=dict(color="darkblue", width=3)))

# === 4. Slider & buttons ===
steps = [
    {
        "args": [[frame.name],
                 {"frame": {"duration": 1500, "redraw": True},  # slower pause on frame
                  "mode": "immediate",
                  "transition": {"duration": 1200, "easing": "cubic-in-out"}}],  # smooth interpolation
        "label": frame.name,
        "method": "animate"
    } for frame in frames
]

sliders = [{
    "steps": steps,
    "transition": {"duration": 1200, "easing": "cubic-in-out"},
    "x": 0.1,
    "y": -0.05,
    "currentvalue": {"prefix": "Year: ", "font": {"size": 16}},
}]

updatemenus = [{
    "buttons": [
        {"args": [None, {"frame": {"duration": 1500, "redraw": True}, "fromcurrent": True}],
         "label": "▶ Play", "method": "animate"},
        {"args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
         "label": "⏸ Pause", "method": "animate"},
    ],
    "direction": "left",
    "x": 0.05,
    "y": -0.1,
    "showactive": False,
}]

# === 5. Layout with initial y-axis ===
fig.update_layout(
    title="Distribution of Sydney House Prices Over Time (2000–2022)",
    xaxis_title="Median House Price ($AUD)",
    yaxis_title="Number of Suburbs (scaled density overlay)",
    bargap=0.05,
    showlegend=False,
    title_font=dict(size=18, family="Arial"),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False, range=[0, year_to_ymax[first_year]]),
    margin=dict(l=40, r=40, t=60, b=80),
    sliders=sliders,
    updatemenus=updatemenus,
    autosize=True,
    template="plotly_white",
)

# === 6. Assign frames for animation ===
fig.frames = frames

# === 7. Show & save ===
fig.show()
fig.write_html("Graph1_sydney_price_distribution_smooth_interpolated.html")