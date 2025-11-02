import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns

results = pd.DataFrame({
    'Model': ['Linear', 'kNN', 'XGboost'],
    'RMSE': [linear_emse, kNN_rmse, xg_rmse],
    'MAE': [linear_mae, kNN_mae, xg_mae],
    'R2': [linear_r2, kNN_r2, xg_r2]
})

plt.figure(figsize=(10,6))
x = np.arange(len(results['Model']))
width = 0.35

plt.bar(x - width/2, results['RMSE'], width, label='RMSE', color='skyblue')
plt.bar(x + width/2, results['MAE'], width, label='MAE', color='salmon')
plt.xticks(x, results['Model'])
plt.title('Comparison of RMSE and MAE Across Models')
plt.ylabel('Error Value')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

for i, v in enumerate(results['RMSE']):
    plt.text(i - 0.2, v + 500, f"{v:.0f}", fontsize=9)
for i, v in enumerate(results['MAE']):
    plt.text(i + 0.1, v + 500, f"{v:.0f}", fontsize=9)
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x='Model', y='R2', data=results, palette='viridis')
plt.title('R² Comparison Across Models')
plt.ylim(0, 1)
plt.ylabel('R² Score')
plt.grid(axis='y', linestyle='--', alpha=0.6)

for i, v in enumerate(results['R2']):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.show()
