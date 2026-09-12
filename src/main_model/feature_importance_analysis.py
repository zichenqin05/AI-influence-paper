"""
特征重要性分析：使用随机森林和排列重要性解释模型决策
分析哪些数据维度对失业风险预测的影响最大

包含：
1. 原始数值特征的直接影响分析
2. 排列重要性（Permutation Importance）
3. SHAP 值可视化（如果 SHAP 库可用）
4. 特征相关性热图
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 数据加载和预处理
# ==========================================
print("📥 加载数据...")
df = pd.read_csv(r"src/embading/merged_ai_impact_jobs.csv")

# 定义失业标签
df['Unemployed'] = 0
if 'NWLK 找工作中' in df.columns:
    df['Unemployed'] = df['Unemployed'] | (df['NWLK 找工作中'] == 1).astype(int)
if 'COW 阶级' in df.columns:
    df['Unemployed'] = df['Unemployed'] | (df['COW 阶级'].isna()).astype(int)

print(f"✅ 失业人数: {df['Unemployed'].sum()} / {len(df)}")

# 提取原始数值特征
numeric_features = [
    "AGEP 年龄",
    "WKWN 过去12个月工作时间",
    "WKL 上次工作时间",
    "SCHL 教育等级",
    "AI_Workload_Ratio"
]

# 分类特征
categorical_features = [
    "NAICSP 行业",
    "SOCP 职位",
]

# 准备特征数据
X_numeric = df[numeric_features].fillna(0)

# 标准化数值特征
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)
X_numeric_scaled = pd.DataFrame(X_numeric_scaled, columns=numeric_features)

y = df['Unemployed'].values

# ==========================================
# 方法1: 随机森林特征重要性
# ==========================================
print("\n" + "="*60)
print("📊 方法1: 随机森林特征重要性分析（基于数值特征）")
print("="*60)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, 
                                  random_state=42, n_jobs=-1, class_weight='balanced')
rf_model.fit(X_numeric_scaled, y)

# 获取特征重要性
feature_importance_rf = pd.DataFrame({
    'Feature': numeric_features,
    'Importance': rf_model.feature_importances_,
    'Importance_Pct': (rf_model.feature_importances_ / rf_model.feature_importances_.sum()) * 100
}).sort_values('Importance', ascending=False)

print("\n🎯 随机森林特征重要性排名:")
print(feature_importance_rf.to_string(index=False))

# 计算累积重要性
feature_importance_rf['Cumulative_Pct'] = feature_importance_rf['Importance_Pct'].cumsum()
print("\n📈 累积重要性:")
print(feature_importance_rf[['Feature', 'Importance_Pct', 'Cumulative_Pct']].to_string(index=False))

# ==========================================
# 方法2: 排列重要性 (Permutation Importance)
# ==========================================
print("\n" + "="*60)
print("📊 方法2: 排列重要性分析")
print("="*60)
print("（衡量打乱特征后模型性能下降程度 → 真实特征贡献度）")

from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(
    rf_model, X_numeric_scaled, y,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

perm_importance_df = pd.DataFrame({
    'Feature': numeric_features,
    'Importance_Mean': perm_importance.importances_mean,
    'Importance_Std': perm_importance.importances_std
}).sort_values('Importance_Mean', ascending=False)

print("\n🎯 排列重要性排名 (Mean ± Std):")
for idx, row in perm_importance_df.iterrows():
    print(f"  {row['Feature']:25s} : {row['Importance_Mean']:8.4f} ± {row['Importance_Std']:.4f}")

# ==========================================
# 方法4: 相关性分析
# ==========================================
print("\n" + "="*60)
print("📊 方法4: 特征与失业标签的相关性")
print("="*60)

corr_data = X_numeric_scaled.copy()
corr_data['Unemployed'] = y

correlation_with_target = corr_data.corr()['Unemployed'][:-1].sort_values(ascending=False)
print("\n🎯 与失业标签的相关系数:")
for feature, corr in correlation_with_target.items():
    print(f"  {feature:25s} : {corr:8.4f}")

# ==========================================
# 方法5: LightGBM 特征重要性（如果模型可用）
# ==========================================
print("\n" + "="*60)
print("📊 方法5: LightGBM 特征重要性（从主 pipeline）")
print("="*60)

try:
    from lightgbm import LGBMClassifier
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    from sentence_transformers import SentenceTransformer
    
    print("🔄 重建 LightGBM 模型以获取特征重要性...")
    
    # 加载 embeddings
    base = Path(__file__).resolve().parent
    embedding_candidates = [
        base / 'job_embeddings.npy',
        base.parent / 'job_embeddings.npy'
    ]
    
    embedding_file = None
    for p in embedding_candidates:
        if p.exists():
            embedding_file = p
            break
    
    if embedding_file:
        embeddings = np.load(embedding_file)
        
        # PCA 降维
        pca = PCA(n_components=50, random_state=42)
        X_pca = pca.fit_transform(embeddings)
        
        # 划分数据集并应用 SMOTE
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, test_size=0.2, random_state=42, stratify=y
        )
        
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # 训练 LightGBM
        lgb_model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=10,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train_smote, y_train_smote)
        
        # 获取特征重要性（前50个PCA维度）
        feature_importance_lgb = lgb_model.feature_importances_
        
        # 计算每个原始特征对PCA维度的贡献
        pca_components_contrib = np.abs(pca.components_)  # (50, embedding_dim)
        
        # 获取前10个最重要的 PCA 分量
        top_pca_idx = np.argsort(feature_importance_lgb)[-10:][::-1]
        
        print("\n🎯 LightGBM 最重要的 PCA 分量 (Top 10):")
        print(f"{'PCA Component':<15} {'Importance':>12} {'Explained Var':>15}")
        print("-" * 45)
        for rank, idx in enumerate(top_pca_idx, 1):
            importance = feature_importance_lgb[idx]
            explained_var = pca.explained_variance_ratio_[idx]
            print(f"PC{idx:<13} {importance:12.4f} {explained_var*100:14.2f}%")
        
        print(f"\n📊 PCA解释的总方差: {pca.explained_variance_ratio_[:50].sum()*100:.2f}%")
        
    else:
        print("⚠️ 未找到 embedding 文件，跳过 LightGBM 分析")
        
except Exception as e:
    print(f"⚠️ LightGBM 分析失败: {e}")

# ==========================================
# 总结
# ==========================================
print("\n" + "="*60)
print("📋 总结：主要影响失业风险的数据维度")
print("="*60)

print("\n🏆 Top 3 最重要的特征 (综合排名):")
print("\n1️⃣ 随机森林重要性:")
for i, row in feature_importance_rf.head(3).iterrows():
    print(f"   {row['Feature']:25s} - {row['Importance_Pct']:.2f}%")

print("\n2️⃣ 排列重要性:")
for i, row in perm_importance_df.head(3).iterrows():
    print(f"   {row['Feature']:25s} - {row['Importance_Mean']:.4f}")

print("\n3️⃣ 特征差异（失业vs在职）:")
for i, row in diff_df.head(3).iterrows():
    print(f"   {row['Feature']:25s} - 差异幅度: {abs(row['Diff_Pct']):.2f}%")

print("\n💡 关键发现:")
print("   - 这些特征对失业风险预测的贡献度最大")
print("   - 可用于：职业风险评估、政策制定、个人规划")
print("   - 建议重点关注前3-5个特征的变化趋势")

print("\n✅ 分析完成！")
