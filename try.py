from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# 创建示例数据集
data = {
    'age': [25, 30, 35, 40, 45, 50, 55, 60],
    'income': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
    'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'target': ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no']
}

df = pd.DataFrame(data)

# 1. 特征和标签分离
X = df[['age', 'income', 'category']]
y = df['target']

# 2. 标签编码
label_encoder = LabelEncoder()
X['category_encoded'] = label_encoder.fit_transform(X['category'])
X = X.drop('category', axis=1)

# 3. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.25, 
    random_state=42, 
    stratify=y
)

print("训练集大小:", X_train.shape)
print("测试集大小:", X_test.shape)

# 5. 训练模型
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. 评估模型
accuracy = model.score(X_test, y_test)
print(f"模型准确率: {accuracy:.2f}")