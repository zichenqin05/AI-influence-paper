"""
完整修复pipeline：数据预处理 → Embedding → 添加失业标签 → 分类模型
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1️⃣ 读取数据并生成失业标签
# ==========================================
print("📥 加载数据...")
df = pd.read_csv(r"src/embading/merged_ai_impact_jobs.csv")

# 🔴 关键：定义失业状态
# COW (阶级) 的含义应该是: 
# - 有值 = 在职
# - 缺失或特殊值 = 失业/不适用
# 同时检查其他字段
df['Unemployed'] = 0  # 默认在职

# 失业条件（选择一个或多个）：
# 1. 如果 NWLK (找工作中) = 1 → 失业
# 2. 如果 COW (阶级) 缺失且 NWLK=1 → 失业
# 3. 如果有特殊标记 → 失业

if 'NWLK 找工作中' in df.columns:
    df['Unemployed'] = df['Unemployed'] | (df['NWLK 找工作中'] == 1).astype(int)

if 'COW 阶级' in df.columns:
    df['Unemployed'] = df['Unemployed'] | (df['COW 阶级'].isna()).astype(int)

print(f"✅ 失业人数: {df['Unemployed'].sum()} / {len(df)}")
print(f"✅ 在职人数: {(df['Unemployed'] == 0).sum()} / {len(df)}")

# ==========================================
# ==========================================
# 2️⃣ 加载或生成 embeddings（单独脚本可避免重复计算）
# ==========================================
print("\n📥 加载 embeddings（优先从 src/embading/job_embeddings.npy 加载）...")
from pathlib import Path
base = Path(__file__).resolve().parent
# 尝试在当前目录和父目录查找 embeddings（处理 copliot 子文件夹情况）
candidate_emb = [base / 'job_embeddings.npy', base.parent / 'job_embeddings.npy']
candidate_meta = [base / 'job_embeddings_meta.csv', base.parent / 'job_embeddings_meta.csv']

EMB_PATH = None
META_PATH = None
for p in candidate_emb:
    if p.exists():
        EMB_PATH = p
        break
for p in candidate_meta:
    if p.exists():
        META_PATH = p
        break

if EMB_PATH is None or META_PATH is None:
    raise FileNotFoundError(
        "找不到 embeddings 文件。请先运行 src/embading/generate_embeddings.py 来生成 job_embeddings.npy 和 job_embeddings_meta.csv"
    )

final_embedding = np.load(EMB_PATH)
meta = pd.read_csv(META_PATH)
print(f"✅ 从已保存文件加载 embeddings，shape={final_embedding.shape} (emb_path={EMB_PATH})")

# 如果 meta 中包含 Unemployed，则使用；否则保持之前生成的 df['Unemployed']
if 'Unemployed' in meta.columns:
    df['Unemployed'] = meta['Unemployed'].fillna(0).astype(int).values


# ==========================================
# 5️⃣ 分类模型
# ==========================================
print("\n🤖 训练分类模型...")

X = final_embedding
y = df['Unemployed'].values

# PCA降维
use_pca = True
pca_dim = 50
if use_pca:
    pca = PCA(n_components=pca_dim, random_state=42)
    X = pca.fit_transform(X)
    print(f"✅ PCA降维: {final_embedding.shape} → {X.shape}")

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE处理类别不平衡
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"✅ SMOTE后训练集: {X_train_res.shape}")

# LightGBM
clf = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    random_state=42,
    verbose=-1
)
clf.fit(X_train_res, y_train_res)

# ==========================================
# 6️⃣ 评估
# ==========================================
print("\n📈 模型评估...")

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['在职', '失业']))
print(f"\nROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# ==========================================
# 7️⃣ 预测在职员工的失业风险
# ==========================================
print("\n🎯 预测在职员工失业风险...")

# 找出训练集中的在职员工
df_employed = df[df['Unemployed'] == 0].copy()
employed_indices = df_employed.index

# 对应的embedding
X_employed = final_embedding[employed_indices]
if use_pca:
    X_employed = pca.transform(X_employed)

# 预测风险评分
risk_scores = clf.predict_proba(X_employed)[:, 1]

df_employed['RiskScore'] = risk_scores
df_employed['Risk_Level'] = pd.cut(
    df_employed['RiskScore'],
    bins=[0, 0.33, 0.66, 1.0],
    labels=['Low', 'Medium', 'High']
)

# 保存
df_employed.to_csv("employed_risk_scores.csv", index=False)

# 统计
print(f"\n高风险员工: {(risk_scores > 0.66).sum()} ({100*(risk_scores > 0.66).mean():.1f}%)")
print(f"中风险员工: {((risk_scores > 0.33) & (risk_scores <= 0.66)).sum()} ({100*((risk_scores > 0.33) & (risk_scores <= 0.66)).mean():.1f}%)")
print(f"低风险员工: {(risk_scores <= 0.33).sum()} ({100*(risk_scores <= 0.33).mean():.1f}%)")

# ==========================================
# 8️⃣ 分析高风险职业
# ==========================================
print("\n🔍 高风险职业分析...")

high_risk_df = df_employed[df_employed['Risk_Level'] == 'High']

if len(high_risk_df) > 0:
    # 职业分布
    job_distribution = high_risk_df['SOCP 职位'].value_counts().head(20)
    print("\n📊 高风险TOP20职业:")
    print(job_distribution)
    
    # 行业分布
    industry_distribution = high_risk_df['NAICSP 行业'].value_counts().head(20)
    print("\n📊 高风险TOP20行业:")
    print(industry_distribution)
    
    # 平均风险评分
    job_avg_risk = high_risk_df.groupby('SOCP 职位')['RiskScore'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(20)
    print("\n📊 风险最高的职业 (按平均评分):")
    print(job_avg_risk)

print("\n✅ 完成！已保存:")
print("  - job_embeddings_with_unemployed_fixed.csv (embedding + 标签)")
print("  - employed_risk_scores.csv (在职员工风险评分)")
