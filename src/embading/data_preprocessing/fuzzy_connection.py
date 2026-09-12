import pandas as pd
from rapidfuzz import process, fuzz

# 1️⃣ 读取数据
df = pd.read_csv(r"src/embading/pure_sampled_data.csv")
ai_df = pd.read_csv(r"src/embading/ai_impact_on_jobs.csv")

# 2️⃣ 取唯一职业（性能关键）
unique_jobs = df['SOCP 职位'].dropna().unique()
ai_jobs = ai_df['Job titiles'].dropna().unique().tolist()

# 3️⃣ 模糊匹配函数
def fuzzy_match_one(x, choices, threshold=50):
    if pd.isna(x):
        return None
    match = process.extractOne(
        x,
        choices,
        scorer=fuzz.token_sort_ratio
    )
    if match and match[1] >= threshold:
        return match[0]
    return None

# 4️⃣ 构造「职业 → AI 职业」映射
job_map = {}
for i, job in enumerate(unique_jobs):
    if i % 20 == 0:
        print(f"Matching job {i}/{len(unique_jobs)}")
    job_map[job] = fuzzy_match_one(job, ai_jobs)

# 5️⃣ 映射回原 df（⚡ 非常快）
df['Occupation_match'] = df['SOCP 职位'].map(job_map)

# 6️⃣ merge AI 影响信息（只按职业）
df = df.merge(
    ai_df[['Job titiles', 'AI Impact', 'Tasks', 'AI models', 'AI_Workload_Ratio']],
    left_on='Occupation_match',
    right_on='Job titiles',
    how='left'
)

# 7️⃣ 清理中间列
df.drop(columns=['Occupation_match', 'Job titiles'], inplace=True)

# 8️⃣ 查看结果
print(df.head())

df = df[df['NAICSP 行业'].notna()]

df.to_csv(r"src/embading/merged_ai_impact_jobs.csv", index=False)