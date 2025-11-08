import pandas as pd
import numpy as np
import webbrowser, os

# === 1. Load dataset ===
df = pd.read_csv("datasets/domain_properties.csv")

# === 2. Parse date and extract year ===
df['date_sold'] = pd.to_datetime(df['date_sold'], dayfirst=True, errors='coerce')
df['year'] = df['date_sold'].dt.year

# Drop rows missing key info and keep only relevant types
df = df.dropna(subset=['price', 'year', 'type'])
df = df[df['type'].isin(['House', 'Apartment / Unit / Flat'])]

# Identify numeric columns (excluding year, suburb, type, price)
num_cols = df.select_dtypes(include=np.number).columns.difference(['year', 'price'])
x_vars = num_cols.tolist()

# === 3. JavaScript interactive plot setup ===
html = f"""
<html>
<head>
    <title>Interactive Price Correlations</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h2 style="text-align:center;">Interactive Correlation Explorer</h2>
    <div style="text-align:center;margin-bottom:10px;">
        <label for="xvar">X-variable:</label>
        <select id="xvar">
            {''.join([f'<option value="{x}">{x}</option>' for x in x_vars])}
        </select>
        <label style="margin-left:20px;">
            <input type="checkbox" id="filterType"> Show only Apartments
        </label>
    </div>
    <div id="plot" style="width:90%;height:80vh;margin:auto;"></div>

    <script>
        let df = {df.to_json(orient='records')};

        function computePlot(xvar, filterApartments) {{
            let x = [], y = [];
            df.forEach(d => {{
                if (d[xvar] != null && d.price != null) {{
                    if (!filterApartments || d.type === 'Apartment / Unit / Flat') {{
                        x.push(d[xvar]);
                        y.push(d.price);
                    }}
                }}
            }});
            if(x.length < 2) {{
                Plotly.newPlot('plot', [], {{title:'Not enough data'}});
                return;
            }}

            // Scatter trace
            let trace = {{
                x: x, y: y, mode: 'markers',
                marker: {{color: 'rgba(0, 100, 255, 0.6)'}},
                name: 'Data'
            }};

            // Linear regression + correlation
            let result = linearRegression(x, y);
            let xfit = linspace(Math.min(...x), Math.max(...x), 100);
            let yfit = xfit.map(xx => result.m * xx + result.b);
            let r = result.r.toFixed(2);
            let line = {{
                x: xfit, y: yfit, mode: 'lines',
                line: {{color: 'darkred', width: 3}},
                name: 'Best Fit (R=' + r + ')'
            }};

            let layout = {{
                title: xvar + ' vs Price ' + (filterApartments ? '(Apartments only)' : '(All types)') + ' — R=' + r,
                xaxis: {{title: xvar}},
                yaxis: {{title: 'Price'}},
                template: 'plotly_white'
            }};

            Plotly.newPlot('plot', [trace, line], layout);
        }}

        function linearRegression(x, y) {{
            let n = x.length;
            let meanX = x.reduce((a,b)=>a+b,0)/n;
            let meanY = y.reduce((a,b)=>a+b,0)/n;
            let num=0, denX=0, denY=0;
            for(let i=0;i<n;i++) {{
                num += (x[i]-meanX)*(y[i]-meanY);
                denX += (x[i]-meanX)**2;
                denY += (y[i]-meanY)**2;
            }}
            let m = num/denX;
            let b = meanY - m*meanX;
            let r = num/Math.sqrt(denX*denY);
            return {{m:m, b:b, r:r}};
        }}

        function linspace(start, stop, num) {{
            let arr = [];
            let step = (stop-start)/(num-1);
            for(let i=0;i<num;i++) arr.push(start + step*i);
            return arr;
        }}

        document.getElementById('xvar').addEventListener('change', () => {{
            computePlot(document.getElementById('xvar').value,
                        document.getElementById('filterType').checked);
        }});
        document.getElementById('filterType').addEventListener('change', () => {{
            computePlot(document.getElementById('xvar').value,
                        document.getElementById('filterType').checked);
        }});

        // Initial plot
        computePlot('{x_vars[0]}', false);
    </script>
</body>
</html>
"""

# Save and open
file_path = "Graph5_interactive_correlation.html"
with open(file_path, "w") as f:
    f.write(html)
print(f"Saved interactive visualization: {file_path}")
webbrowser.open("file://" + os.path.abspath(file_path))