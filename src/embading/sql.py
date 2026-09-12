import pymysql
import pandas as pd

conn = pymysql.connect(host =  'localhost',
                      user = 'root',
                      password = '20051108',
                      db = '2026UsaMo',
                          port = 3306,
                          charset = 'utf8'
                      )

#sql基本语句
def do(oder):
    cursor = conn.cursor()
    cursor.execute(oder)
    result = cursor.fetchall()
    cursor.close()
    return result

#sql查询前20条数据
def order_top20(action_type):

    query = f"""
    SELECT 
        `1` AS product,
        `2` AS category,
        COUNT(*) AS {action_type}_count
    FROM 
        sampled_data
    WHERE 
        `3` = '{action_type}'
    GROUP BY 
        `1`, `2`
    ORDER BY 
        {action_type}_count DESC
    LIMIT 20;
    """
    
    # 执行查询
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    
    # 转为DataFrame并保存CSV
    df = pd.DataFrame(result, columns=['product', 'category', f'{action_type}_count'])
    csv_filename = f'top20_{action_type}.csv'
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    print(f"已保存: {csv_filename}")
    return df

import pandas as pd

def join_query(tables, join_conditions, select_columns, limit=10):
    # 生成SELECT部分
    select_clause = ", ".join(select_columns)
    
    # 生成FROM和JOIN部分
    from_clause = f"{tables[0][0]} {tables[0][1]}"  # 第一个表
    for i in range(1, len(tables)):
        table_name, alias = tables[i]
        condition = join_conditions[i-1]
        from_clause += f"\nINNER JOIN {table_name} {alias} ON {condition}"
    
    # 生成完整SQL
    sql = f"""
    SELECT {select_clause}
    FROM {from_clause}
    LIMIT {limit};
    """
    
    # 执行查询（使用已有的conn连接）
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        
        # 获取列名（用于DataFrame表头）
        columns = [desc[0] for desc in cursor.description]
        
        # 获取结果并转为DataFrame
        result = cursor.fetchall()
        df = pd.DataFrame(result, columns=columns)
        
        cursor.close()
        return df
        
    except Exception as e:
        print(f"查询出错: {str(e)}")
        print("执行的SQL语句:")
        print(sql)
        return None

def search(type, table):
    query = (f"""SELECT 
        {type},
        COUNT(*) AS count
    FROM 
        {table}
    GROUP BY 
        {type};""")

    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    
    df = pd.DataFrame(result, columns=[f'{type}', 'count'])

    return df