import pandas as pd

# 读取数据
file_path = 'D://YOLO//ultralytics-main//test_camera//dis_data//data.xlsx'
df = pd.read_excel(file_path)

# 计算 X 与 Z 的相关性
correlation_xz = df[['X', 'Z']].corr().iloc[0, 1]

# 计算 Y 与 Z 的相关性
correlation_yz = df[['Y', 'Z']].corr().iloc[0, 1]

print(f"Correlation between X and Z: {correlation_xz:.4f}")
print(f"Correlation between Y and Z: {correlation_yz:.4f}")