import pandas as pd
import plotly.express as px
import json
import webbrowser, os

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

# === 3. Median Price figure ===
fig_median = px.choropleth_map(
    df_grouped,
    geojson=geojson_data,
    locations="suburb",
    featureidkey=f"properties.{suburb_key}",
    color="median",
    hover_name="suburb",
    hover_data={"SydneyRegion": True, "median": ":.0f", "yoy_change": ":.1f"},
    animation_frame="year",
    map_style="carto-positron",
    color_continuous_scale="YlOrRd",
    range_color=(df_grouped["median"].quantile(0.05),
                 df_grouped["median"].quantile(0.95)),
    title="Sydney Median House Prices by Suburb (Animated by Year)",
    center={"lat": -33.8688, "lon": 151.2093},
    zoom=9,
    opacity=0.75,
)

# === 4. YoY Change figure ===
fig_yoy = px.choropleth_map(
    df_grouped,
    geojson=geojson_data,
    locations="suburb",
    featureidkey=f"properties.{suburb_key}",
    color="yoy_change",
    hover_name="suburb",
    hover_data={"SydneyRegion": True, "median": ":.0f", "yoy_change": ":.1f"},
    animation_frame="year",
    map_style="carto-positron",
    color_continuous_scale="RdBu",
    range_color=(-20, 20),
    title="Sydney YoY Change in Median House Prices (%)",
    center={"lat": -33.8688, "lon": 151.2093},
    zoom=9,
    opacity=0.75,
)

# === 5. Save both figures to HTML and embed toggle controls ===
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

with open("Graph3_sydney_price_heatmap_toggle.html", "w", encoding="utf-8") as f:
    f.write(html_template)

print("✅ Saved: Graph3_sydney_price_heatmap_toggle.html")
webbrowser.open("file://" + os.path.abspath("Graph3_sydney_price_heatmap_toggle.html"))