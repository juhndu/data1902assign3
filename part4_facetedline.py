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