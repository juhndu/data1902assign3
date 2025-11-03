import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['MLR', 'KNN', 'SVM', 'XGB']
rmse = [59323.61, 62638.54, 59975.48, 55091.57]
mae = [42003.10, 43019.3, 41267.29, 37781.24]
r2 = [0.647, 0.608, 0.64, 0.697]

# Set up figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [1, 1.2]})
bar_width = 0.35
x = np.arange(len(models))

# --- Top chart: R² values ---
ax1.bar(models, r2, color='skyblue', edgecolor='black')
ax1.set_title('Model R² Values', fontsize=12, weight='bold')
ax1.set_ylabel('R²', fontsize=11)
ax1.set_ylim(0, 1)
for i, val in enumerate(r2):
    ax1.text(i, val + 0.02, f'{val:.3f}', ha='center', fontsize=9)
ax1.grid(axis='y', linestyle='--', alpha=0.6)

# --- Bottom chart: RMSE and MAE (clustered bar chart) ---
ax2.bar(x - bar_width/2, rmse, bar_width, label='RMSE', color='orange', edgecolor='black')
ax2.bar(x + bar_width/2, mae, bar_width, label='MAE', color='lightgreen', edgecolor='black')

ax2.set_title('Model RMSE and MAE Comparison', fontsize=12, weight='bold')
ax2.set_ylabel('Error Value')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.6)

# --- Add subtitle below the figure ---
plt.figtext(0.5, 0.02, 
            "Figure 11: Comparison of average R² (top) and RMSE/MAE (bottom) for four predictive models on the test data", 
            ha="center", fontsize=10, style='italic')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()