# -----------------------------------------
# Sydney House Price Distribution Over Time
# Animated Histogram + Smooth Density Curve
# -----------------------------------------
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



# -----------------------------------------
# Graph 2: Median House Price Trends by Sydney Region (Faceted Line Chart)
# -----------------------------------------
import pandas as pd
import plotly.express as px

# === 1. Load and clean dataset ===
df = pd.read_csv("processed_data/suburb_yearly_bybednum_sydney_enriched.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Ensure proper data types
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["median"] = pd.to_numeric(df["median"], errors="coerce")

# Drop missing or invalid values
df = df.dropna(subset=["year", "median", "SydneyRegion"])

# Keep data in reasonable range (avoid outliers)
df = df[(df["median"] > 100000) & (df["median"] < 5_000_000)]

# Ensure years are sorted correctly
years_sorted = sorted(df["year"].unique())
df["year"] = pd.Categorical(df["year"], categories=years_sorted, ordered=True)

print(f"Dataset ready — {df['SydneyRegion'].nunique()} regions, Years {years_sorted[0]}–{years_sorted[-1]}")

# === 2. Aggregate to regional level (median per year per region) ===
region_df = (
    df.groupby(["SydneyRegion", "year"])["median"]
    .median()
    .reset_index()
)

# === 3. Create faceted line chart ===
fig = px.line(
    region_df,
    x="year",
    y="median",
    color="SydneyRegion",
    facet_col="SydneyRegion",
    facet_col_wrap=4,  # arrange in grid
    title="Median House Prices Over Time by Sydney Region (2000–2022)",
    labels={"median": "Median Price ($AUD)", "year": "Year"},
    color_discrete_sequence=px.colors.qualitative.Set2,
)

# === 4. Style and layout ===
fig.update_traces(mode="lines+markers", line=dict(width=2))

fig.update_layout(
    height=950,
    showlegend=False,
    title_font=dict(size=20, family="Arial"),
    plot_bgcolor="white",
    margin=dict(l=40, r=40, t=60, b=40),
)

# Add gridline + axis formatting
fig.for_each_xaxis(lambda axis: axis.update(showgrid=False, tickangle=45))
fig.for_each_yaxis(lambda axis: axis.update(showgrid=False))

# === 5. Display and export ===
fig.show()
fig.write_html("Graph2_sydney_region_trends_faceted.html")

print("Saved faceted line chart as HTML.")



# -----------------------------------------
# Sydney Median Price Animated Heatmap (Fixed Version)
# -----------------------------------------
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json, os, webbrowser

# === 1. Load and prepare your data ===
df = pd.read_csv("processed_data/suburb_yearly_bybednum_sydney_enriched.csv")
df.columns = df.columns.str.strip()
df["suburb"] = df["suburb"].str.strip().str.upper()
df["SydneyRegion"] = df["SydneyRegion"].str.strip()
df = df[["suburb", "year", "median", "SydneyRegion"]].dropna()

df_grouped = (
    df.groupby(["suburb", "year", "SydneyRegion"])["median"]
    .median()
    .reset_index()
)
df_grouped["year"] = df_grouped["year"].astype(int)
df_grouped = df_grouped.sort_values("year")
df_grouped = df_grouped[df_grouped["year"] != df_grouped["year"].max()]
df_grouped["yoy_change"] = (
    df_grouped.groupby("suburb")["median"].pct_change() * 100
).replace([float("inf"), -float("inf")], None).fillna(0)

# === 2. Load GeoJSON ===
geo_path = "suburb-10-nsw.geojson"
with open(geo_path) as f:
    geojson_data = json.load(f)

suburb_key = "nsw_loca_2"
suburb_list = df_grouped["suburb"].unique()
geojson_data["features"] = [
    f for f in geojson_data["features"]
    if f["properties"].get(suburb_key, "").upper() in suburb_list
]

# === 3. Region label coordinates ===
region_coords = {
    "Eastern Suburbs": (-33.90, 151.27),
    "Lower North Shore": (-33.82, 151.22),
    "Upper North Shore": (-33.71, 151.12),
    "Northern Beaches": (-33.70, 151.30),
    "North Shore": (-33.76, 151.15),
    "Northern Suburbs": (-33.78, 151.10),
    "Inner West": (-33.89, 151.14),
    "Inner South": (-33.95, 151.17),
    "South West": (-33.96, 150.90),
    "Western Suburbs": (-33.85, 150.95),
    "Sydney City": (-33.86, 151.21),
    "Hills Shire": (-33.72, 150.97),
    "Sutherland Shire": (-34.03, 151.05),
    "Southern Suburbs": (-33.98, 151.12),
}

# === 4. Create helper to add region labels ===
def add_region_labels(fig):
    """Add text overlays for 14 major Sydney regions."""
    for region, (lat, lon) in region_coords.items():
        fig.add_trace(go.Scattermapbox(
            lat=[lat],
            lon=[lon],
            mode="text",
            text=[f"{region}"],
            textfont=dict(size=13, color="dark gray"),
            textposition="middle center",
            hoverinfo="skip",
            showlegend=False,
        ))
    return fig

# === 5. Median Price Map ===
fig_median = px.choropleth_mapbox(
    df_grouped,
    geojson=geojson_data,
    locations="suburb",
    featureidkey=f"properties.{suburb_key}",
    color="median",
    hover_name="suburb",
    hover_data={"SydneyRegion": True, "median": ":.0f", "yoy_change": ":.1f"},
    animation_frame="year",
    mapbox_style="carto-positron",
    color_continuous_scale="YlOrRd",
    range_color=(df_grouped["median"].quantile(0.05),
                 df_grouped["median"].quantile(0.95)),
    title="Sydney Median House Prices by Suburb (Animated by Year)",
    center={"lat": -33.8688, "lon": 151.2093},
    zoom=9,
    opacity=0.75,
)
fig_median = add_region_labels(fig_median)

# === 6. YoY Change Map ===
fig_yoy = px.choropleth_mapbox(
    df_grouped,
    geojson=geojson_data,
    locations="suburb",
    featureidkey=f"properties.{suburb_key}",
    color="yoy_change",
    hover_name="suburb",
    hover_data={"SydneyRegion": True, "median": ":.0f", "yoy_change": ":.1f"},
    animation_frame="year",
    mapbox_style="carto-positron",
    color_continuous_scale="RdBu",
    range_color=(-20, 20),
    title="Sydney YoY Change in Median House Prices (%)",
    center={"lat": -33.8688, "lon": 151.2093},
    zoom=9,
    opacity=0.75,
)
fig_yoy = add_region_labels(fig_yoy)

# === 7. Save both figures to HTML with toggle buttons ===
html_template = f"""
<html>
<head>
    <title>Sydney House Prices</title>
    <style>
        body {{ margin: 0; background: #fff; padding-top: 20px; }}
        .button-container {{
            position: fixed;
            top: 10px;
            left: 80px;
            z-index: 100;
        }}
        button {{
            margin-right: 10px;
            padding: 8px 14px;
            font-size: 14px;
            border: 1px solid #555;
            border-radius: 4px;
            background-color: #f7f7f7;
            cursor: pointer;
        }}
        button:hover {{
            background-color: #e0e0e0;
        }}
    </style>
</head>
<body>
    <div class="button-container">
        <button onclick="showMedian()">Median Price</button>
        <button onclick="showYoY()">YoY Change (%)</button>
    </div>
    <div id="median" style="display:block">{fig_median.to_html(include_plotlyjs='cdn', full_html=False)}</div>
    <div id="yoy" style="display:none">{fig_yoy.to_html(include_plotlyjs=False, full_html=False)}</div>
    <script>
        function showMedian() {{
            document.getElementById('median').style.display = 'block';
            document.getElementById('yoy').style.display = 'none';
        }}
        function showYoY() {{
            document.getElementById('median').style.display = 'none';
            document.getElementById('yoy').style.display = 'block';
            window.dispatchEvent(new Event('resize'));
        }}
    </script>
</body>
</html>
"""

output_path = "Graph3_sydney_price_heatmap_toggle.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_template)

print(f"Saved: {output_path}")
webbrowser.open("file://" + os.path.abspath(output_path))


# -----------------------------------------
# NSW Property Data — Animated Boxplot of Price per m² by Region (Ordered Years)
# -----------------------------------------
import pandas as pd
import plotly.express as px
import numpy as np

# === 1. Load dataset ===
df = pd.read_csv("processed_data/nsw_propertydata_sydney_cleaned.csv")

# === 2. Clean columns ===
df.columns = df.columns.str.strip()
df["purchase_price"] = pd.to_numeric(df["purchase_price"], errors="coerce")
df["area"] = pd.to_numeric(df["area"], errors="coerce")
df = df.dropna(subset=["purchase_price", "area", "SydneyRegion", "download_date"])
df = df[(df["purchase_price"] > 0) & (df["area"] > 10)]

# === 3. Extract year and compute price per m² ===
df["year"] = pd.to_datetime(df["download_date"], errors="coerce").dt.year
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)
df["price_per_sqm"] = df["purchase_price"] / df["area"]

# === 4. Trim extreme outliers (optional but helps readability) ===
df = df[(df["price_per_sqm"] > 500) & (df["price_per_sqm"] < 50000)]

# === 5. Combine Lower + Upper North Shore for display only ===
region_map = {
    "Lower North Shore": "North Shore",
    "Upper North Shore": "North Shore"
}
df["SydneyRegionDisplay"] = df["SydneyRegion"].replace(region_map)

# === 6. Chronological order for year (use string for animation) ===
years_sorted = sorted(df["year"].unique().tolist())
years_sorted_str = [str(y) for y in years_sorted]
df["year_str"] = df["year"].astype(str)

# === 7. Custom region order (merged version for display) ===
region_order_display = [
    "Sydney City",
    "Eastern Suburbs",
    "Inner East",
    "Inner South",
    "Inner West",
    "North Shore",            # merged
    "Northern Suburbs",
    "Northern Beaches",
    "Southern Suburbs",
    "Sutherland Shire",
    "Hills Shire",
    "Western Suburbs",
    "South West",
]
df["SydneyRegionDisplay"] = pd.Categorical(df["SydneyRegionDisplay"], categories=region_order_display, ordered=True)

print(f"Data ready — {len(df)} records, {len(years_sorted)} years, {df['SydneyRegionDisplay'].nunique()} display regions")

# === 8. Animated boxplot ===
fig = px.box(
    df,
    x="SydneyRegionDisplay",
    y="price_per_sqm",
    animation_frame="year_str",                 # use ordered string years
    color="SydneyRegionDisplay",
    points="outliers",
    title="Sydney Property Space-for-Money Trade-off (AUD per m²) — Animated by Year",
    labels={
        "SydneyRegionDisplay": "Region",
        "price_per_sqm": "Price per Square Metre ($AUD/m²)",
        "year_str": "Year",
    },
    color_discrete_sequence=px.colors.qualitative.Set2,
    category_orders={                           # FORCE ordering
        "year_str": years_sorted_str,
        "SydneyRegionDisplay": region_order_display,
    },
)

# --- Extra guard: explicitly sort frames by year ---
fig.frames = tuple(sorted(fig.frames, key=lambda fr: int(fr.name)))

# === 9. Layout improvements ===
fig.update_layout(
    height=750,
    title_font=dict(size=20, family="Arial", color="black"),
    xaxis=dict(title="Sydney Region", tickangle=25, categoryorder="array", categoryarray=region_order_display),
    yaxis=dict(title="Price per Square Metre ($AUD/m²)", showgrid=True),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=60, r=60, t=80, b=140),
    showlegend=False,
)

# Keep y-range consistent across years for smoother animation
fig.update_yaxes(range=[0, df["price_per_sqm"].quantile(0.98)])

# === 10. Display and save ===
fig.show()
fig.write_html("Graph4_animated_boxplot_price_per_sqm_merged_northshore.html")

print("Saved: Graph4_animated_boxplot_price_per_sqm_merged_northshore.html")