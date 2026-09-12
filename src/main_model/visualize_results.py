"""
读取 `employed_risk_scores.csv` 并在控制台显示三张美化的柱状图：
 - 风险最高的职业 (按平均评分)
 - 高风险职业（按数量）
 - 高风险行业（按数量，行业名简化）

本脚本直接 `plt.show()` 显示图表，不保存文件。
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 美化配置
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

ROOT = Path(__file__).resolve().parent
# 支持 copliot 子目录：优先在当前目录查找，其次查找父目录
candidate = [ROOT / 'employed_risk_scores.csv', ROOT.parent / 'employed_risk_scores.csv']
IN_CSV = None
for p in candidate:
    if p.exists():
        IN_CSV = p
        break

if IN_CSV is None:
    print(f"文件不存在：在 {candidate} 均未找到 employed_risk_scores.csv，请先运行 pipeline 生成该文件")
    raise SystemExit(1)

print("📥 加载风险评分数据...")
df = pd.read_csv(IN_CSV)

# 确保 RiskScore 为数值
df['RiskScore'] = pd.to_numeric(df['RiskScore'], errors='coerce')

# 简化行业名称：去掉前缀 (如 "MFG-...")，并截短
def simplify_industry(name):
    if pd.isna(name):
        return 'NA'
    s = str(name)
    if '-' in s:
        s = s.split('-', 1)[1]
    return s.strip()[:40]

if 'NAICSP 行业' in df.columns:
    df['NAICSP_short'] = df['NAICSP 行业'].apply(simplify_industry)

# 1) 风险最高的职业（按平均评分）
if 'SOCP 职位' in df.columns:
    job_avg = df.groupby('SOCP 职位')['RiskScore'].mean().dropna().sort_values(ascending=False).head(20)
    plt.figure(figsize=(10,6))
    sns.barplot(x=job_avg.values, y=job_avg.index, palette='viridis')
    plt.title('Top20 Jobs by Average RiskScore')
    plt.xlabel('Average RiskScore')
    plt.xlim(0,1)
    for i, v in enumerate(job_avg.values):
        plt.text(v + 0.01, i, f"{v:.2f}", va='center')
    plt.tight_layout()
    plt.show()
    print('\nTop20 Jobs by Average RiskScore:')
    print(job_avg)

# 2) 高风险职业（按数量）
if 'SOCP 职位' in df.columns:
    high_jobs = df[df['RiskScore']>0.66]['SOCP 职位'].value_counts().head(20)
    plt.figure(figsize=(10,6))
    sns.barplot(x=high_jobs.values, y=high_jobs.index, palette='rocket')
    plt.title('Top20 High-Risk Jobs (count)')
    plt.xlabel('Count')
    for i, v in enumerate(high_jobs.values):
        plt.text(v + max(high_jobs.values)*0.01, i, str(v), va='center')
    plt.tight_layout()
    plt.show()
    print('\nTop20 High-Risk Jobs (count):')
    print(high_jobs)

# 3) 高风险行业（按数量，使用简化名称）
if 'NAICSP_short' in df.columns:
    top_inds = df[df['RiskScore']>0.66]['NAICSP_short'].value_counts().head(20)
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_inds.values, y=top_inds.index, palette='mako')
    plt.title('Top20 High-Risk Industries (count)')
    plt.xlabel('Count')
    for i, v in enumerate(top_inds.values):
        plt.text(v + max(top_inds.values)*0.01, i, str(v), va='center')
    plt.tight_layout()
    plt.show()
    print('\nTop20 High-Risk Industries (count):')
    print(top_inds)

# 4) 最不容易被AI替代的职业（RiskScore最低）
if 'SOCP 职位' in df.columns:
    job_min = df.groupby('SOCP 职位')['RiskScore'].mean().dropna().sort_values(ascending=True).head(20)
    plt.figure(figsize=(10,6))
    sns.barplot(x=job_min.values, y=job_min.index, palette='cool')
    plt.title('Top20 Jobs Least Likely to be Replaced by AI')
    plt.xlabel('Average RiskScore (Lower = Safer)')
    plt.xlim(0,1)
    for i, v in enumerate(job_min.values):
        plt.text(v + 0.01, i, f"{v:.2f}", va='center')
    plt.tight_layout()
    plt.show()
    print('\nTop20 Jobs Least Likely to be Replaced by AI:')
    print(job_min)

print('\n🎉 可视化完成（已在窗口显示）')
