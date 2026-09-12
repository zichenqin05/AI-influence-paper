import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

# ======================
# 1️⃣ 读取 CSV
# ======================
df = pd.read_csv(r"src\embading\merged_ai_impact_jobs.csv")

# 去掉 NAICSP 或 SOCP 为空的行
df = df.dropna(subset=["NAICSP 行业", "SOCP 职位"]).reset_index(drop=True)

# ======================
# 2️⃣ 数值特征和有序类别
# ======================
numeric_cols = [
    "AGEP 年龄",
    "WKWN 过去12个月工作时间",
    "WKL 上次工作时间",
    "AI_Workload_Ratio"
]

ordinal_cols = [
    "SCHL 教育等级"
]

# 填充缺失值
numeric_data = df[numeric_cols + ordinal_cols].fillna(0)

# 标准化
scaler = StandardScaler()
numeric_emb = scaler.fit_transform(numeric_data)

# ======================
# 3️⃣ 文本特征 embedding
# ======================
text_cols = ["NAICSP 行业", "SOCP 职位", "Tasks", "AI models"]

# 拼接文本
df["text_for_embedding"] = (
    df["NAICSP 行业"].fillna("").astype(str) + " | " +
    df["SOCP 职位"].fillna("").astype(str) + " | Tasks: " +
    df["Tasks"].fillna("").astype(str) + " | AI models: " +
    df["AI models"].fillna("").astype(str)
)

# 加载模型
model = SentenceTransformer("all-MiniLM-L6-v2")

# 生成 embedding
text_emb = model.encode(df["text_for_embedding"].tolist(), show_progress_bar=True)

# ======================
# 4️⃣ 拼接最终 embedding
# ======================
final_embedding = np.hstack([numeric_emb, text_emb])

print("Final embedding shape:", final_embedding.shape)
# 每一行就是一个向量，对应一条样本

# ======================
# 5️⃣ 保存 embedding
# ======================
# 保存为 numpy 文件
np.save("job_embeddings.npy", final_embedding)

# 如果想要 CSV，每行是向量
np.savetxt("job_embeddings.csv", final_embedding, delimiter=",")
