"""
Visualize WKL distribution for unemployed population
Creates a beautiful bar chart showing percentages
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data: WKL categories with counts and percentages
data = {
    'WKL_Category': [1, 2, 3],
    'Count': [3164, 1025, 939],
    'Percent': [61.700, 19.988, 18.311]
}

# WKL category descriptions
wkl_labels = {
    1: '1: Within the past 12 months',
    2: '2: 1-5 years ago',
    3: '3: Over 5 years ago or never worked'
}

df = pd.DataFrame(data)
df['Label'] = df['WKL_Category'].map(wkl_labels)

# Set up visualization style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.titlesize': 16,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Create bar chart with gradient colors
colors = sns.color_palette('viridis', len(df))
bars = ax.bar(range(len(df)), df['Percent'], color=colors, edgecolor='black', linewidth=1.5)

# Customize axes
ax.set_xlabel('Last Work Time Category (WKL)', fontsize=13, fontweight='bold')
ax.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold')
ax.set_title('Unemployed: When Last Work', fontsize=16, fontweight='bold')
ax.set_ylim(0, 100)

# Set x-axis labels with descriptions
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df['Label'], fontsize=11)

# Add value labels on top of bars with percentages and counts
for i, (bar, val) in enumerate(zip(bars, df['Percent'])):
    height = bar.get_height()
    count = df.iloc[i]['Count']
    ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
            f'{val:.2f}%\n(n={count})',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add grid for better readability
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("Chart displayed successfully!")
print(f"\nData Summary:")
for idx, row in df.iterrows():
    print(f"{row['Label']}: {row['Percent']:.2f}% (n={row['Count']})")
print(f"\nTotal: {df['Percent'].sum():.2f}% (n={df['Count'].sum()})")
