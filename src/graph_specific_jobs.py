import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# 设置中文字体（可选，如需显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# Data (English only)
# =========================
jobs = [
    "Food & Service\nOperations",
    "Motion & Video\nDesign",
    "Chemistry\nEngineering",
]

people = np.array([569, 24, 3])
risk = np.array([0.68, 0.29, 0.05])

# =========================
# Create figure with professional style
# =========================
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('#FFFFFF')

# =========================
# Create grouped bar chart with enhanced styling
# =========================
x = np.arange(len(jobs))
width = 0.36

# Bars for number of people (left axis)
bars1 = ax1.bar(x - width/2, people, width, label='Number of People',
                color='#3498DB', edgecolor='#2C3E50', linewidth=2, alpha=0.9)

# Add gradient-like effect by varying alpha
for bar in bars1:
    bar.set_linewidth(2.5)

# Create second y-axis for risk
ax2 = ax1.twinx()

# Bars for risk (right axis as percentage)
risk_pct = risk * 100  # Convert to percentage (0-100)
bars2 = ax2.bar(x + width/2, risk_pct, width, label='AI Replacement Risk (%)',
                color='#E74C3C', edgecolor='#C0392B', linewidth=2, alpha=0.9)

for bar in bars2:
    bar.set_linewidth(2.5)

# =========================
# Axis styling with colors
# =========================
ax1.set_xlabel('Occupational Category', fontsize=13, fontweight='bold', 
               labelpad=12, color='#2C3E50')
ax1.set_ylabel('Number of People', fontsize=13, fontweight='bold', 
               labelpad=12, color='#3498DB')
ax2.set_ylabel('AI Replacement Risk (%)', fontsize=13, fontweight='bold', 
               labelpad=12, color='#E74C3C')

# Set x-axis ticks and labels
ax1.set_xticks(x)
ax1.set_xticklabels(jobs, fontsize=12, fontweight='bold', color='#2C3E50')

# Customize y-axis ticks and colors
ax1.tick_params(axis='y', labelcolor='#3498DB', labelsize=11, width=1.5, length=6)
ax2.tick_params(axis='y', labelcolor='#E74C3C', labelsize=11, width=1.5, length=6)
ax1.tick_params(axis='x', labelsize=12, length=6)

# Set grid style
ax1.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.8, color='#BDC3C7')
ax1.set_axisbelow(True)

# =========================
# Spine styling
# =========================
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(2.5)
ax1.spines['left'].set_color('#3498DB')
ax1.spines['bottom'].set_linewidth(2)
ax1.spines['bottom'].set_color('#2C3E50')

ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_linewidth(2.5)
ax2.spines['right'].set_color('#E74C3C')

# =========================
# Set Y-axis limits with better spacing
# =========================
ax1.set_ylim(0, 700)
ax2.set_ylim(0, 100)

# =========================
# Title and subtitle
# =========================
fig.suptitle('Selected AI Job Replacement Risk Analysis', 
             fontsize=18, fontweight='bold', y=0.98, color='#2C3E50')

# =========================
# Data annotations with improved formatting
# =========================
for i, (p, r) in enumerate(zip(people, risk)):
    # People count annotation - positioned above blue bars
    ax1.text(i - width/2, p + 25, f'{int(p)}',
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            color='#3498DB', family='monospace')
    
    # Risk percentage annotation - positioned above red bars
    r_pct = r * 100
    ax2.text(i + width/2, r_pct + 3, f'{r_pct:.0f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            color='#E74C3C', family='monospace')

# =========================
# Create custom legend with better styling and positioning
# =========================
blue_patch = mpatches.Patch(color='#3498DB', edgecolor='#2C3E50', linewidth=2, 
                           label='Number of People')
red_patch = mpatches.Patch(color='#E74C3C', edgecolor='#C0392B', linewidth=2, 
                          label='AI Replacement Risk (%)')

legend = ax1.legend(handles=[blue_patch, red_patch],
                   loc='upper right', fontsize=12, framealpha=0.98,
                   edgecolor='#2C3E50', fancybox=True, shadow=True,
                   frameon=True, bbox_to_anchor=(0.98, 0.97))

# Improve legend appearance
legend.get_frame().set_linewidth(1.5)
legend.get_frame().set_facecolor('#FFFFFF')



plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92)
plt.show()
