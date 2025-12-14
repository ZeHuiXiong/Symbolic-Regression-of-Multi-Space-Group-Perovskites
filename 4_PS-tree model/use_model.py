import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor
from gplearn.genetic import SymbolicRegressor

# 必须定义与训练时相同的 PSTree 类
class PSTree:
    def __init__(self, n_partitions=4, max_depth=3, population_size=1000, generations=50):
        self.n_partitions = n_partitions
        self.max_depth = max_depth
        self.population_size = population_size
        self.generations = generations
        self.tree_model = None
        self.symbolic_models = {}
        self.sample_counts = {}  # 记录分区样本数
        self.expressions = {}  # 存储表达式字符串

    def fit(self, X, y):
        # 决策树分区
        self.tree_model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=max(10, len(X) // (self.n_partitions * 10)),
            random_state=42
        )
        self.tree_model.fit(X, y)

        leaves = self.tree_model.apply(X)
        unique_leaves = np.unique(leaves)

        # 分区符号回归
        for leaf in unique_leaves:
            mask = (leaves == leaf)
            X_leaf = X[mask]
            y_leaf = y[mask]

            if len(X_leaf) < 10:
                continue

            sr = SymbolicRegressor(
                population_size=self.population_size,
                generations=self.generations,
                tournament_size=20,
                stopping_criteria=0.01,
                const_range=(-1.0, 1.0),
                init_depth=(2, 6),
                function_set=(
                'add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan'),
                parsimony_coefficient=0.001,
                random_state=42,
                verbose=0
            )
            sr.fit(X_leaf.values, y_leaf.values)  # 输入numpy数组

            self.symbolic_models[leaf] = sr
            self.sample_counts[leaf] = len(X_leaf)  # 记录样本数
            self.expressions[leaf] = str(sr._program)  # 直接存储表达式

    def predict(self, X):
        predictions = np.zeros(len(X))
        leaves = self.tree_model.apply(X)

        for i, leaf in enumerate(leaves):
            if leaf in self.symbolic_models:
                pred_array = self.symbolic_models[leaf].predict(X.iloc[i:i + 1, :])
                predictions[i] = pred_array.item()  # 显式提取标量
            else:
                global_avg = np.mean([m.predict(X.iloc[:1, :])[0] for m in self.symbolic_models.values()])
                predictions[i] = global_avg

        return predictions

# 加载模型和数据集
model_path = "/Users/mrbear/PycharmProjects/pythonProject/Work_No4/FHHG/PS-Tree/ps_tree_model.pkl"
data_path = "/Users/mrbear/PycharmProjects/pythonProject/Work_No4/Model/NEW/2step/final-key5/7176_all_key5.csv"

# 加载模型
ps_tree = joblib.load(model_path)

# 加载数据集
df = pd.read_csv(data_path)
feature_cols = ['D_max','EN_max','r_B2X', 'avg_dev Electronegativity', 'mean NdValence']
target_col = 'band_gap'

# 确保数据集包含所有需要的列
required_cols = feature_cols + [target_col]
if not set(required_cols).issubset(df.columns):
    missing = set(required_cols) - set(df.columns)
    raise ValueError(f"数据集缺少必要的列: {missing}")

# 提取特征和目标
X = df[feature_cols]
y = df[target_col]

# 使用模型进行预测
print("开始进行预测...")
y_pred = ps_tree.predict(X)
print("预测完成!")

# 计算绝对误差
abs_error = np.abs(y - y_pred)

# 创建结果数据框
results = pd.DataFrame({
    'Actual': y,
    'Predicted': y_pred,
    'Absolute_Error': abs_error
})

# 添加原始数据索引以便追踪
results.reset_index(inplace=True, drop=False)
results.rename(columns={'index': 'Original_Index'}, inplace=True)

# 保存结果到CSV
output_path = "/Users/mrbear/PycharmProjects/pythonProject/Work_No4/FHHG/PS-Tree/prediction_results.csv"
results.to_csv(output_path, index=False)
print(f"预测结果已保存到: {output_path}")
print(f"总样本数: {len(results)}")
print(f"平均绝对误差: {abs_error.mean():.6f}")