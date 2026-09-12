"""
特征重要性统计可视化脚本
展示Top 10特征的4个关键统计图表
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import pearsonr, spearmanr

# 美化配置
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

# ==========================================
# 数据加载和特征准备
# ==========================================
print("="*80)
print("📊 TOP 10 FEATURE IMPORTANCE VISUALIZATION")
print("="*80)

print("\n📥 Loading data...")
df = pd.read_csv(r"src/embading/merged_ai_impact_jobs.csv", encoding='gbk')

# 定义失业标签
df['Unemployed'] = 0
if 'NWLK 找工作中' in df.columns:
    df['Unemployed'] = df['Unemployed'] | (df['NWLK 找工作中'] == 1).astype(int)
if 'COW 阶级' in df.columns:
    df['Unemployed'] = df['Unemployed'] | (df['COW 阶级'].isna()).astype(int)

# 原始数值特征 - 去除与失业相关和过去12个月工作事件的变量
# NWAB(临时缺勤)、NWLA(裁员)、NWLK(找工作中)、NWRE(召回) 应作为Y变量
# WKWN(过去12个月工作时间)、NWAV(可供工作使用) 不能作为特征
numeric_features_cn = [
    "AGEP 年龄",
    "SCHL 教育等级",
    "AI_Workload_Ratio",
    "OIP 其他收入",
    "PAP 救济收入",
    "DIS 残疾",
    "SEX 性别"
]

# 中文到英文的映射
feature_name_mapping = {
    "AGEP 年龄": "Age",
    "SCHL 教育等级": "Education_Level",
    "AI_Workload_Ratio": "AI_Workload_Ratio",
    "OIP 其他收入": "Other_Income",
    "PAP 救济收入": "Relief_Income",
    "DIS 残疾": "Disability",
    "SEX 性别": "Gender"
}

numeric_features = [feature_name_mapping[f] for f in numeric_features_cn]

# 准备特征数据
X_numeric = df[numeric_features_cn].fillna(0)
X_numeric.columns = numeric_features
X_numeric = X_numeric.replace([np.inf, -np.inf], 0)

y = df['Unemployed'].values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)
X_scaled = pd.DataFrame(X_scaled, columns=numeric_features)

print(f"✅ Features: {X_scaled.shape[1]} | Samples: {X_scaled.shape[0]}")
print(f"✅ Unemployed: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}")

# ==========================================
# 特征与目标变量相关性分析 - 验证特征的统计显著性
# ==========================================
print("\n" + "="*80)
print("🔍 FEATURE CORRELATION WITH TARGET VARIABLE")
print("="*80)

correlation_results = []
for feature in numeric_features:
    # Pearson相关性
    pearson_corr, p_value_pearson = pearsonr(X_scaled[feature], y)
    # Spearman相关性（排名相关）
    spearman_corr, p_value_spearman = spearmanr(X_scaled[feature], y)
    
    correlation_results.append({
        'Feature': feature,
        'Pearson_Corr': pearson_corr,
        'Pearson_P': p_value_pearson,
        'Spearman_Corr': spearman_corr,
        'Spearman_P': p_value_spearman,
        'Significant': 'Yes' if min(p_value_pearson, p_value_spearman) < 0.05 else 'No'
    })

corr_df = pd.DataFrame(correlation_results).sort_values('Pearson_Corr', key=abs, ascending=False)
print("\n📊 Correlation Analysis (with p-values):")
print(corr_df.to_string(index=False))

print("\n" + "="*80)
print("\n🔨 Training Random Forest Model...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)

# MDI特征重要性
feature_importance_mdi = pd.DataFrame({
    'Feature': numeric_features,
    'Importance': rf_model.feature_importances_,
    'Importance_Pct': (rf_model.feature_importances_ / rf_model.feature_importances_.sum()) * 100
}).sort_values('Importance', ascending=False)

top10_mdi = feature_importance_mdi.head(10)
print(f"✅ Random Forest MDI Importance computed (Top 10 shown)")

# ==========================================
# 随机森林模型性能评估
# ==========================================
print("\n" + "="*80)
print("📊 RANDOM FOREST MODEL PERFORMANCE")
print("="*80)

rf_y_pred_train = rf_model.predict(X_train)
rf_y_pred_test = rf_model.predict(X_test)
rf_y_prob_test = rf_model.predict_proba(X_test)[:, 1]

print("\n🎯 Training Set Metrics:")
print(f"   Accuracy:  {accuracy_score(y_train, rf_y_pred_train):.4f}")
print(f"   Precision: {precision_score(y_train, rf_y_pred_train, zero_division=0):.4f}")
print(f"   Recall:    {recall_score(y_train, rf_y_pred_train, zero_division=0):.4f}")
print(f"   F1-Score:  {f1_score(y_train, rf_y_pred_train, zero_division=0):.4f}")

print("\n🎯 Test Set Metrics:")
print(f"   Accuracy:  {accuracy_score(y_test, rf_y_pred_test):.4f}")
print(f"   Precision: {precision_score(y_test, rf_y_pred_test, zero_division=0):.4f}")
print(f"   Recall:    {recall_score(y_test, rf_y_pred_test, zero_division=0):.4f}")
print(f"   F1-Score:  {f1_score(y_test, rf_y_pred_test, zero_division=0):.4f}")
print(f"   ROC-AUC:   {roc_auc_score(y_test, rf_y_prob_test):.4f}")

print("\n📋 Confusion Matrix (Test Set):")
cm = confusion_matrix(y_test, rf_y_pred_test)
print(f"   TP: {cm[1,1]:4d} | FP: {cm[0,1]:4d}")
print(f"   FN: {cm[1,0]:4d} | TN: {cm[0,0]:4d}")

print("\n📊 Classification Report (Test Set):")
print(classification_report(y_test, rf_y_pred_test, target_names=['在职', '失业'], digits=4))

# ==========================================
# 2. 计算排列重要性 - 测试集
# ==========================================
print("\n🔄 Computing Permutation Importance on Test Set...")
perm_importance_test = permutation_importance(
    rf_model, X_test, y_test,
    n_repeats=15,
    random_state=42,
    n_jobs=-1,
    scoring='f1'
)

perm_importance_df = pd.DataFrame({
    'Feature': numeric_features,
    'Importance_Mean': perm_importance_test.importances_mean,
    'Importance_Std': perm_importance_test.importances_std
}).sort_values('Importance_Mean', ascending=False)

top10_perm = perm_importance_df.head(10)
print(f"✅ Permutation Importance computed (Top 10 shown)")

# ==========================================
# 交叉验证特征重要性稳定性
# ==========================================
print("\n" + "="*80)
print("✅ CROSS-VALIDATION STABILITY CHECK")
print("="*80)

from sklearn.model_selection import StratifiedKFold

cv_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_importances = []

for fold, (train_idx, val_idx) in enumerate(cv_fold.split(X_scaled, y), 1):
    X_cv_train, X_cv_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
    y_cv_train, y_cv_val = y[train_idx], y[val_idx]
    
    rf_cv = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_cv.fit(X_cv_train, y_cv_train)
    cv_importances.append(rf_cv.feature_importances_)

cv_importances = np.array(cv_importances)
cv_mean = cv_importances.mean(axis=0)
cv_std = cv_importances.std(axis=0)

cv_stability = pd.DataFrame({
    'Feature': numeric_features,
    'Mean_Importance': cv_mean,
    'Std_Importance': cv_std,
    'Stability_CV': (1 - cv_std / (cv_mean + 1e-6))  # 稳定性得分，越接近1越稳定
}).sort_values('Mean_Importance', ascending=False)

print("\n5-Fold Cross-Validation Stability:")
print(cv_stability.to_string(index=False))

print(f"\n🎯 Average Feature Importance Std Dev: {cv_std.mean():.4f}")
print("   (Lower std = More Stable = More Reliable)")

# ==========================================
# 3. 计算排列重要性 - 测试集
# ==========================================
print("\n🔄 Computing Permutation Importance on Test Set...")

# ==========================================
# 3. 计算累积重要性 - 基于MDI
# ==========================================
print("\n📊 Computing Cumulative Importance...")
cumsum_importance = feature_importance_mdi.copy()
cumsum_importance['Cumulative_Pct'] = cumsum_importance['Importance_Pct'].cumsum()
# ensure main MDI dataframe also exposes cumulative percentage for plotting later
feature_importance_mdi['Cumulative_Pct'] = feature_importance_mdi['Importance_Pct'].cumsum()
top10_cumsum = cumsum_importance.head(10)
print(f"✅ Cumulative Importance computed")

# ==========================================
# 4. LightGBM + PCA - 最重要的分量
# ==========================================
print("\n🚀 Training LightGBM with PCA Components...")

# 加载embeddings
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

lightgbm_data = None
if embedding_file:
    try:
        embeddings = np.load(embedding_file)
        
        # PCA降维
        pca = PCA(n_components=50, random_state=42)
        X_pca = pca.fit_transform(embeddings)
        
        # 划分数据并应用SMOTE
        X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
            X_pca, y, test_size=0.2, random_state=42, stratify=y
        )
        
        smote = SMOTE(random_state=42)
        X_train_pca_smote, y_train_pca_smote = smote.fit_resample(X_train_pca, y_train_pca)
        
        # 训练LightGBM
        lgb_model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=10,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train_pca_smote, y_train_pca_smote)
        
        # ==========================================
        # LightGBM 模型性能评估
        # ==========================================
        print("\n" + "="*80)
        print("📊 LIGHTGBM MODEL PERFORMANCE (PCA + SMOTE)")
        print("="*80)
        
        lgb_y_pred_train = lgb_model.predict(X_train_pca)
        lgb_y_pred_test = lgb_model.predict(X_test_pca)
        lgb_y_prob_test = lgb_model.predict_proba(X_test_pca)[:, 1]
        
        print("\n🎯 Training Set Metrics:")
        print(f"   Accuracy:  {accuracy_score(y_train_pca, lgb_y_pred_train):.4f}")
        print(f"   Precision: {precision_score(y_train_pca, lgb_y_pred_train, zero_division=0):.4f}")
        print(f"   Recall:    {recall_score(y_train_pca, lgb_y_pred_train, zero_division=0):.4f}")
        print(f"   F1-Score:  {f1_score(y_train_pca, lgb_y_pred_train, zero_division=0):.4f}")
        
        print("\n🎯 Test Set Metrics:")
        print(f"   Accuracy:  {accuracy_score(y_test_pca, lgb_y_pred_test):.4f}")
        print(f"   Precision: {precision_score(y_test_pca, lgb_y_pred_test, zero_division=0):.4f}")
        print(f"   Recall:    {recall_score(y_test_pca, lgb_y_pred_test, zero_division=0):.4f}")
        print(f"   F1-Score:  {f1_score(y_test_pca, lgb_y_pred_test, zero_division=0):.4f}")
        print(f"   ROC-AUC:   {roc_auc_score(y_test_pca, lgb_y_prob_test):.4f}")
        
        print("\n📋 Confusion Matrix (Test Set):")
        cm_lgb = confusion_matrix(y_test_pca, lgb_y_pred_test)
        print(f"   TP: {cm_lgb[1,1]:4d} | FP: {cm_lgb[0,1]:4d}")
        print(f"   FN: {cm_lgb[1,0]:4d} | TN: {cm_lgb[0,0]:4d}")
        
        print("\n📊 Classification Report (Test Set):")
        print(classification_report(y_test_pca, lgb_y_pred_test, target_names=['在职', '失业'], digits=4))
        
        # 获取特征重要性
        feature_importance_lgb = lgb_model.feature_importances_
        top_n = 10
        top_indices = np.argsort(feature_importance_lgb)[-top_n:][::-1]
        
        lightgbm_data = pd.DataFrame({
            'PCA_Component': [f'PC{i}' for i in top_indices],
            'Importance': feature_importance_lgb[top_indices],
            'Explained_Var': pca.explained_variance_ratio_[top_indices] * 100
        })
        
        print(f"✅ LightGBM model trained | PCA top 10 components extracted")
        
        pca_cumsum = np.cumsum(pca.explained_variance_ratio_) * 100
        
    except Exception as e:
        print(f"⚠️ LightGBM processing failed: {e}")
        lightgbm_data = None
else:
    print("⚠️ Embedding file not found, skipping LightGBM analysis")

# ==========================================
# 图1: 随机森林特征重要性排名 - Top 10
# ==========================================
print("\n📈 Generating Visualizations...")

plt.figure(figsize=(12, 7))
colors_mdi = sns.color_palette('viridis', len(top10_mdi))
bars = plt.barh(range(len(top10_mdi)), top10_mdi['Importance_Pct'].values[::-1], color=colors_mdi[::-1])
plt.yticks(range(len(top10_mdi)), top10_mdi['Feature'].values[::-1], fontsize=11)
plt.xlabel('Importance (%)', fontsize=13, fontweight='bold')
plt.ylabel('Features', fontsize=13, fontweight='bold')
plt.title('Top 10 Features - Random Forest Importance (MDI)', fontsize=15, fontweight='bold')
plt.xlim(0, max(top10_mdi['Importance_Pct']) * 1.15)

for i, v in enumerate(top10_mdi['Importance_Pct'].values[::-1]):
    plt.text(v + 0.3, i, f'{v:.2f}%', va='center', fontsize=10, fontweight='bold')

plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("✅ Plot 1: Random Forest Feature Importance - Top 10")

# ==========================================
# 图2: 排列重要性排名 - Top 10 (Mean ± Std)
# ==========================================
plt.figure(figsize=(12, 7))
colors_perm = sns.color_palette('rocket', len(top10_perm))
x_pos = range(len(top10_perm))

plt.barh(x_pos, top10_perm['Importance_Mean'].values[::-1], 
         xerr=top10_perm['Importance_Std'].values[::-1],
         color=colors_perm[::-1], capsize=6, error_kw={'linewidth': 2, 'ecolor': 'black'})

plt.yticks(x_pos, top10_perm['Feature'].values[::-1], fontsize=11)
plt.xlabel('Permutation Importance (Mean ± Std)', fontsize=13, fontweight='bold')
plt.ylabel('Features', fontsize=13, fontweight='bold')
plt.title('Top 10 Features - Permutation Importance (Test Set)', fontsize=15, fontweight='bold')

max_val = (top10_perm['Importance_Mean'] + top10_perm['Importance_Std']).max()
plt.xlim(0, max_val * 1.15)

for i, (mean, std) in enumerate(zip(top10_perm['Importance_Mean'].values[::-1], 
                                     top10_perm['Importance_Std'].values[::-1])):
    plt.text(mean + std + 0.01, i, f'{mean:.4f}±{std:.4f}', va='center', fontsize=9, fontweight='bold')

plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("✅ Plot 2: Permutation Importance (Mean ± Std) - Top 10")

# ==========================================
# 图3: 累积重要性 - Top 10
# ==========================================
fig, ax = plt.subplots(figsize=(12, 7))

# 上方条形图：个别重要性
top10_cumsum_sorted = top10_cumsum.sort_values('Importance_Pct', ascending=False)
colors_indiv = sns.color_palette('husl', len(top10_cumsum_sorted))

x_pos = range(len(top10_cumsum_sorted))
bars = ax.bar(x_pos, top10_cumsum_sorted['Importance_Pct'].values, 
              color=colors_indiv, label='Individual Importance', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Individual Importance (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Features (Ranked)', fontsize=13, fontweight='bold')
ax.set_title('Top 10 Features - Individual & Cumulative Importance', fontsize=15, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(top10_cumsum_sorted['Feature'].values, rotation=45, ha='right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# 添加个别重要性的数值标签
for i, (bar, val) in enumerate(zip(bars, top10_cumsum_sorted['Importance_Pct'].values)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# 在图表上方添加累积百分比信息
cumsum_values = top10_cumsum_sorted['Cumulative_Pct'].values
for i, cumsum_val in enumerate(cumsum_values):
    ax.text(i, max(top10_cumsum_sorted['Importance_Pct']) * 0.95, f'Cum: {cumsum_val:.1f}%',
            ha='center', va='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.show()

# 单独显示累积曲线
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(len(feature_importance_mdi)), feature_importance_mdi['Cumulative_Pct'].values, 
        'o-', linewidth=3, markersize=8, color='#e74c3c', label='Cumulative Importance')
ax.axhline(y=80, color='orange', linestyle='--', linewidth=2, label='80% Threshold')
ax.axhline(y=90, color='red', linestyle='--', linewidth=2, label='90% Threshold')
ax.fill_between(range(len(feature_importance_mdi)), feature_importance_mdi['Cumulative_Pct'].values,
                alpha=0.2, color='#e74c3c')

ax.set_xlabel('Number of Features', fontsize=13, fontweight='bold')
ax.set_ylabel('Cumulative Importance (%)', fontsize=13, fontweight='bold')
ax.set_title('Cumulative Importance Curve - All Features', fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, loc='lower right')
ax.set_xlim(0, len(feature_importance_mdi)-1)
ax.set_ylim(0, 105)

plt.tight_layout()
plt.show()

print("✅ Plot 3: Cumulative Importance - Top 10 & Curve")

# ==========================================
# 图4: LightGBM 最重要的 PCA 分量 - Top 10
# ==========================================
if lightgbm_data is not None:
    fig, ax = plt.subplots(figsize=(12, 7))
    colors_lgb = sns.color_palette('coolwarm', len(lightgbm_data))
    
    bars = ax.barh(range(len(lightgbm_data)), lightgbm_data['Importance'].values[::-1], 
                   color=colors_lgb[::-1], edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(range(len(lightgbm_data)))
    ax.set_yticklabels(lightgbm_data['PCA_Component'].values[::-1], fontsize=11)
    ax.set_xlabel('LightGBM Feature Importance', fontsize=13, fontweight='bold')
    ax.set_ylabel('PCA Components', fontsize=13, fontweight='bold')
    ax.set_title('Top 10 PCA Components by LightGBM Importance', fontsize=15, fontweight='bold')
    ax.set_xlim(0, max(lightgbm_data['Importance']) * 1.15)
    
    # 添加重要性数值和解释方差
    for i, (imp, exp_var) in enumerate(zip(lightgbm_data['Importance'].values[::-1],
                                            lightgbm_data['Explained_Var'].values[::-1])):
        ax.text(imp + 0.5, i, f'{imp:.1f} (var: {exp_var:.1f}%)', 
                va='center', fontsize=9, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    print("✅ Plot 4: LightGBM Top 10 PCA Components")
    
    # 额外：PCA解释方差累积曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    cumsum_pca = np.cumsum(pca.explained_variance_ratio_) * 100
    
    ax.plot(range(1, 51), cumsum_pca, 'o-', linewidth=3, markersize=7, color='#2ecc71', label='Cumulative Variance')
    ax.axhline(y=80, color='orange', linestyle='--', linewidth=2, label='80% Threshold')
    ax.axhline(y=90, color='red', linestyle='--', linewidth=2, label='90% Threshold')
    ax.fill_between(range(1, 51), cumsum_pca, alpha=0.2, color='#2ecc71')
    
    # 标记top 10
    top_components = np.argsort(lightgbm_data['Importance'].values)[-10:][::-1] + 1
    for comp in top_components[:5]:  # 只标记前5个
        idx = comp - 1
        ax.scatter(comp, cumsum_pca[idx], s=200, color='red', zorder=5)
        ax.annotate(f'PC{comp}', xy=(comp, cumsum_pca[idx]), xytext=(comp+1, cumsum_pca[idx]+3),
                   fontsize=9, fontweight='bold', arrowprops=dict(arrowstyle='->', color='red'))
    
    ax.set_xlabel('Number of PCA Components', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Explained Variance (%)', fontsize=13, fontweight='bold')
    ax.set_title('PCA Cumulative Explained Variance - All 50 Components', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Plot 5: PCA Cumulative Variance Curve (Bonus)")
else:
    print("⚠️ LightGBM data not available, skipping plot 4")

# ==========================================
# 总结输出
# ==========================================
print("\n" + "="*80)
print("📋 SUMMARY - TOP 10 FEATURES")
print("="*80)

print("\n🏆 Random Forest MDI - Top 10:")
for rank, (idx, row) in enumerate(top10_mdi.iterrows(), 1):
    print(f"   {rank:2d}. {row['Feature']:25s} {row['Importance_Pct']:7.2f}%")

print("\n🎯 Permutation Importance - Top 10 (Test Set):")
for rank, (idx, row) in enumerate(top10_perm.iterrows(), 1):
    print(f"   {rank:2d}. {row['Feature']:25s} {row['Importance_Mean']:8.4f} ± {row['Importance_Std']:.4f}")

# ==========================================
# 特征重要性一致性验证
# ==========================================
print("\n" + "="*80)
print("🔗 FEATURE IMPORTANCE CONSISTENCY CHECK")
print("="*80)

# 将三种方法的排名进行标准化对比
mdi_ranking = pd.DataFrame({
    'Feature': top10_mdi['Feature'].values,
    'MDI_Rank': range(1, len(top10_mdi)+1)
})

perm_ranking = pd.DataFrame({
    'Feature': top10_perm['Feature'].values,
    'Perm_Rank': range(1, len(top10_perm)+1)
})

cv_ranking = pd.DataFrame({
    'Feature': cv_stability.head(10)['Feature'].values,
    'CV_Rank': range(1, 11)
})

# 合并排名
consistency_check = mdi_ranking.copy()
consistency_check = consistency_check.merge(perm_ranking, on='Feature', how='left')
consistency_check = consistency_check.merge(cv_ranking, on='Feature', how='left')

# 计算排名标准差
consistency_check['Rank_Std'] = consistency_check[['MDI_Rank', 'Perm_Rank', 'CV_Rank']].std(axis=1, skipna=True)
consistency_check = consistency_check.sort_values('Rank_Std')

print("\n📊 Top 10 Feature - Ranking Consistency Across Methods:")
print("   (Lower Rank_Std = Better Consistency = More Reliable)")
print(consistency_check.to_string(index=False))

# 统计有多少特征排名一致
high_consistency = (consistency_check['Rank_Std'] <= 2).sum()
print(f"\n✅ High Consistency Features (Std ≤ 2): {high_consistency}/10")
print(f"🔶 Medium Consistency Features (2 < Std ≤ 4): {((consistency_check['Rank_Std'] > 2) & (consistency_check['Rank_Std'] <= 4)).sum()}/10")
print(f"⚠️  Low Consistency Features (Std > 4): {(consistency_check['Rank_Std'] > 4).sum()}/10")

print("\n" + "="*80)
print("📊 MODEL ACCURACY COMPARISON")
print("="*80)

print("\n🤖 Random Forest (Baseline):")
print(f"   Test Accuracy:  {accuracy_score(y_test, rf_y_pred_test):.4f}")
print(f"   Test F1-Score:  {f1_score(y_test, rf_y_pred_test, zero_division=0):.4f}")
print(f"   Test ROC-AUC:   {roc_auc_score(y_test, rf_y_prob_test):.4f}")

if lightgbm_data is not None:
    print(f"\n🚀 LightGBM (PCA + SMOTE):")
    print(f"   Test Accuracy:  {accuracy_score(y_test_pca, lgb_y_pred_test):.4f}")
    print(f"   Test F1-Score:  {f1_score(y_test_pca, lgb_y_pred_test, zero_division=0):.4f}")
    print(f"   Test ROC-AUC:   {roc_auc_score(y_test_pca, lgb_y_prob_test):.4f}")

print("\n✅ Visualization Complete!")
print("="*80)
