# 🔴 模型问题诊断与修复方案

## 📋 问题分析

### 1️⃣ **核心问题：失业标签缺失**

你的原pipeline流程：
```
data_pure.py (文本映射) → main_embading.py (生成embedding) → main.py (训练分类器)
```

**问题**：`main_embading.py` 生成的embedding**没有包含失业标签**，但 `main.py` 期望有 `Unemployed` 列。

**错误链路**：
```python
# main.py 第15行期望：
df2 = pd.read_csv(r"job_embeddings_with_unemployed.csv")  # ❌ 这个文件不存在！
y = df2['Unemployed'].to_numpy()  # ❌ 因为该列从未被创建
```

---

## 🔧 修复方案

### 步骤1：定义失业状态（关键）

在数据中，失业应该通过以下字段识别：
```python
df['Unemployed'] = 0  # 默认在职

# 条件1：如果"找工作中"=1，则失业
if 'NWLK 找工作中' in df.columns:
    df['Unemployed'] = df['Unemployed'] | (df['NWLK 找工作中'] == 1).astype(int)

# 条件2：如果"阶级"(COW)缺失，则可能失业
if 'COW 阶级' in df.columns:
    df['Unemployed'] = df['Unemployed'] | (df['COW 阶级'].isna()).astype(int)
```

### 步骤2：生成完整embedding + 标签

使用新的 `pipeline_fixed.py`：
```bash
python src/embading/pipeline_fixed.py
```

**输出文件**：
- `job_embeddings_with_unemployed_fixed.csv` - embedding + 失业标签
- `employed_risk_scores.csv` - 在职员工的失业风险评分

---

## 🎯 你的逻辑流程修正

### 原本想法：
> 先将数据embed成向量 → 在向量空间中包含已失业和未失业的人 → 用分类模型预测未失业的人的失业风险 → 回查职业

### ✅ 修正后的实现：

```
┌─────────────────────────────────────────────────┐
│ 1. 数据预处理 + 失业标签                         │
│    - 从原始数据识别失业人员                      │
│    - 标记: Unemployed = 0 (在职) / 1 (失业)    │
└────────────┬────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────┐
│ 2. 特征化 + Embedding                           │
│    - 数值特征 (年龄、工作时间等) + 标准化      │
│    - 文本特征 (职业、行业、AI任务等) + BERT   │
│    - 拼接: [数值emb, 文本emb]                  │
└────────────┬────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────┐
│ 3. 分类模型训练                                 │
│    - 使用 LightGBM 训练在 embedding 空间      │
│    - 已失业(1) vs 在职(0)                      │
│    - 用 SMOTE 处理类别不平衡                   │
└────────────┬────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────┐
│ 4. 风险预测                                     │
│    - 对所有在职员工(Unemployed=0)预测         │
│    - 输出 RiskScore ∈ [0, 1]                   │
│    - 高分 = 高失业风险                          │
└────────────┬────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────┐
│ 5. 职业聚合分析                                 │
│    - 按职业分组统计高风险员工                   │
│    - 识别 TOP 风险职业                          │
│    - 统计这些职业的未来走向                     │
└─────────────────────────────────────────────────┘
```

---

## 📊 预期输出示例

运行 `pipeline_fixed.py` 后：

```
✅ 失业人数: 5230 / 45000
✅ 在职人数: 39770 / 45000

✅ 模型评估
分类报告:
                 precision    recall  f1-score   support
              在职       0.92      0.88      0.90      7954
              失业       0.65      0.72      0.68      1846
              
ROC-AUC: 0.8234

🎯 失业风险分布
高风险员工: 3210 (8.1%)
中风险员工: 7890 (19.8%)
低风险员工: 28670 (72.1%)

📊 高风险TOP10职业:
  1. CMS-Mental Health Counselors - 245 人 (平均风险: 0.78)
  2. ENG-Biomedical Engineers - 187 人 (平均风险: 0.76)
  3. CMM-Web Developers - 156 人 (平均风险: 0.74)
  ...
```

---

## 🚀 运行步骤

### 1. 清理旧文件
```bash
rm job_embeddings.npy
rm job_embeddings.csv
rm lightgbm_risk_scores.csv
```

### 2. 运行新pipeline
```bash
cd src/embading
python pipeline_fixed.py
```

### 3. 分析结果
```bash
# 查看高风险职业
cat employed_risk_scores.csv | sort -t',' -k'RiskScore' -rn | head -20
```

---

## ⚠️ 常见问题

### Q: 为什么失业人数那么少？
**A**: 检查你的原始数据中失业的定义。可能需要调整：
```python
# 尝试这个替代条件
df['Unemployed'] = (df['NWLK 找工作中'] == 1) | \
                   (df['NWAB 临时缺勤'] == 1) | \
                   (df['NWLA 裁员'] == 1)
```

### Q: 模型准确率太低？
**A**: 可能原因：
1. 失业标签定义不准确
2. embedding维度太高 → 调整 `pca_dim`
3. 类别比例太不平衡 → 调整 SMOTE 参数

### Q: 如何验证逻辑正确？
**A**: 检查 `employed_risk_scores.csv` 中：
- 是否所有行的 `Unemployed` 都是 0？
- 风险分数分布是否合理（不全是0或1）？
- 高风险职业是否符合直觉（如高AI冲击职业）？

---

## 📚 参考

- NAICS: 行业分类
- SOCP: 职业分类  
- COW: Class of Worker（阶级）
- NWLK: Looking for work (是否在找工作)
