"""
生成并保存 embeddings（单独运行以减少重复计算）
输出：
 - src/embading/job_embeddings.npy
 - src/embading/job_embeddings_meta.csv
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(__file__)
CSV_PATH = os.path.join(ROOT, "merged_ai_impact_jobs.csv")
OUT_EMB = os.path.join(ROOT, "job_embeddings.npy")
OUT_META = os.path.join(ROOT, "job_embeddings_meta.csv")

print("📥 加载数据...")
df = pd.read_csv(CSV_PATH)

# 保留原始索引以便对齐
meta = df.copy()

# 数值特征
numeric_cols = [
    "AGEP 年龄",
    "WKWN 过去12个月工作时间",
    "WKL 上次工作时间"
]
if 'AI_Workload_Ratio' in df.columns:
    numeric_cols.append('AI_Workload_Ratio')
ordinal_cols = ["SCHL 教育等级"]

numeric_data = df[numeric_cols + ordinal_cols].fillna(0)
numeric_data = numeric_data.replace([np.inf, -np.inf], 0)

scaler = StandardScaler()
numeric_emb = scaler.fit_transform(numeric_data)

# 文本特征
text_cols = ["NAICSP 行业", "SOCP 职位"]
if "Tasks" in df.columns:
    text_cols.append("Tasks")
if "AI models" in df.columns:
    text_cols.append("AI models")

print("🎯 生成文本embedding（使用 sentence-transformers）...")
df["text_for_embedding"] = ""
for col in text_cols:
    df["text_for_embedding"] += df[col].fillna("").astype(str) + " | "

model = SentenceTransformer("all-MiniLM-L6-v2")
text_emb = model.encode(df["text_for_embedding"].tolist(), show_progress_bar=True)

# 拼接
final_embedding = np.hstack([numeric_emb, text_emb])
print("✅ final embedding shape:", final_embedding.shape)

# 保存
np.save(OUT_EMB, final_embedding)
meta.to_csv(OUT_META, index=False, encoding='utf-8-sig')
print(f"✅ Saved embeddings -> {OUT_EMB}")
print(f"✅ Saved meta -> {OUT_META}")
