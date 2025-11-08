# -----------------------------------------
# NSW Property Data — Animated Boxplot of Price per m² by Region (Ordered Years)
# -----------------------------------------
import pandas as pd
import plotly.express as px
import numpy as np

# === 1. Load dataset ===
df = pd.read_csv("datasets/nsw_propertydata_sydney_cleaned.csv")

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


