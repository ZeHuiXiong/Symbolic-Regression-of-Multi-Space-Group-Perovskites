import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
import os
import textwrap

# 设置中文字体（确保系统已安装）
plt.rcParams['font.family'] = 'Heiti TC'

# 数据加载
file_path = "/Users/mrbear/PycharmProjects/pythonProject/Work_No4/Model/NEW/2step/final-key6/7176_key6_final.csv"  # 替换实际路径
df = pd.read_csv(file_path)
feature_cols = ['D_max', 'EG_max', 'r_B2X', 'EG_B2', 'mean NdValence', 'EG_B1', ]
target_col = 'band_gap'
data = df[feature_cols + [target_col]]

# 数据集划分
X = data[feature_cols]
y = data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


class PSTree:
    def __init__(self, tree_params=None, sr_params=None):
        """
        初始化PS-Tree模型

        参数:
        tree_params (dict): 决策树分区参数
            - n_partitions (int): 期望的分区数量 (默认: 4)
            - max_depth (int): 决策树最大深度 (默认: 5)
            - min_samples_leaf (int): 叶节点最小样本数 (默认: 自动计算)

        sr_params (dict): 符号回归参数
            - population_size (int): 种群大小 (默认: 1000)
            - generations (int): 进化代数 (默认: 50)
            - tournament_size (int): 锦标赛选择大小 (默认: 20)
            - stopping_criteria (float): 停止标准 (默认: 0.01)
            - const_range (tuple): 常数范围 (默认: (-1.0, 1.0))
            - init_depth (tuple): 初始深度范围 (默认: (2, 6))
            - function_set (tuple): 函数集 (默认: 基本数学函数)
            - parsimony_coefficient (float): 简约系数 (默认: 0.001)
            - metric (str): 评估指标 (默认: 'rmse')
        """
        # 设置决策树默认参数
        self.tree_params = tree_params or {}
        self.tree_params.setdefault('n_partitions', 4)
        self.tree_params.setdefault('max_depth', 5)

        # 设置符号回归默认参数
        self.sr_params = sr_params or {}
        self.sr_params.setdefault('population_size', 1000)
        self.sr_params.setdefault('generations', 50)
        self.sr_params.setdefault('tournament_size', 20)
        self.sr_params.setdefault('stopping_criteria', 0.01)
        self.sr_params.setdefault('const_range', (-1.0, 1.0))
        self.sr_params.setdefault('init_depth', (2, 6))
        self.sr_params.setdefault('function_set', ('add', 'sub', 'mul', 'div', 'sqrt', 'log',
                                                   'abs', 'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan'))
        self.sr_params.setdefault('parsimony_coefficient', 0.001)
        self.sr_params.setdefault('metric', 'rmse')

        # 模型组件
        self.tree_model = None
        self.symbolic_models = {}
        self.sample_counts = {}
        self.expressions = {}
        self.histories = {}  # 存储每个分区的训练历史
        self.partition_data = {}  # 存储每个分区的数据

    def fit(self, X, y):
        """训练PS-Tree模型"""
        # 计算决策树的最小叶节点样本数
        n_partitions = self.tree_params['n_partitions']
        min_samples_leaf = max(10, len(X) // (n_partitions * 10))

        # 创建决策树模型
        self.tree_model = DecisionTreeRegressor(
            max_depth=self.tree_params['max_depth'],
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        self.tree_model.fit(X, y)

        leaves = self.tree_model.apply(X)
        unique_leaves = np.unique(leaves)

        print(f"\n开始分区符号回归训练（共{len(unique_leaves)}个分区）...")
        print(f"符号回归参数: 种群大小={self.sr_params['population_size']}, 代数={self.sr_params['generations']}")

        with tqdm(total=len(unique_leaves), desc="训练进度") as pbar:
            for leaf in unique_leaves:
                mask = (leaves == leaf)
                X_leaf = X[mask]
                y_leaf = y[mask]

                # 保存分区数据
                partition_df = pd.concat([X_leaf, y_leaf], axis=1)
                self.partition_data[leaf] = partition_df

                if len(X_leaf) < 10:
                    pbar.update(1)
                    continue

                # 配置 SymbolicRegressor
                sr = SymbolicRegressor(
                    population_size=self.sr_params['population_size'],
                    generations=self.sr_params['generations'],
                    tournament_size=self.sr_params['tournament_size'],
                    stopping_criteria=self.sr_params['stopping_criteria'],
                    const_range=self.sr_params['const_range'],
                    init_depth=self.sr_params['init_depth'],
                    function_set=self.sr_params['function_set'],
                    parsimony_coefficient=self.sr_params['parsimony_coefficient'],
                    random_state=42,
                    verbose=0,  # 关闭原生输出避免冲突
                    metric=self.sr_params['metric']
                )

                sr.fit(X_leaf.values, y_leaf.values)

                # 记录结果
                self.symbolic_models[leaf] = sr
                self.sample_counts[leaf] = len(X_leaf)
                self.expressions[leaf] = str(sr._program)
                self.histories[leaf] = sr.run_details_  # 原生历史记录

                pbar.set_postfix({
                    f"分区{leaf}": f"{len(X_leaf)}样本",
                    "最佳RMSE": f"{min(sr.run_details_['best_fitness']):.4f}"
                })
                pbar.update(1)

    def predict(self, X):
        """使用训练好的模型进行预测"""
        predictions = np.zeros(len(X))
        leaves = self.tree_model.apply(X)

        for i, leaf in enumerate(leaves):
            if leaf in self.symbolic_models:
                pred_array = self.symbolic_models[leaf].predict(X.iloc[i:i + 1, :])
                predictions[i] = pred_array.item()
            else:
                # 如果遇到未知分区，使用所有模型的平均预测值
                global_avg = np.mean([m.predict(X.iloc[:1, :])[0] for m in self.symbolic_models.values()])
                predictions[i] = global_avg

        return predictions

    # 可视化训练历史（使用原生数据）
    def plot_training_history(self, save_dir="training_history"):
        """绘制每个分区的训练历史图"""
        os.makedirs(save_dir, exist_ok=True)

        for leaf, history in self.histories.items():
            plt.figure(figsize=(12, 5))

            # 适应度曲线
            plt.subplot(1, 2, 1)
            plt.plot(history['generation'], history['best_fitness'], 'b-')
            plt.title(f'分区 {leaf} - 适应度进化')
            plt.xlabel('代数')
            plt.ylabel('RMSE')
            plt.grid(True)

            # 复杂度曲线
            plt.subplot(1, 2, 2)
            plt.plot(history['generation'], history['best_length'], 'r-')
            plt.title(f'分区 {leaf} - 表达式复杂度')
            plt.xlabel('代数')
            plt.ylabel('节点数')
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f"{save_dir}/partition_{leaf}_history.png", dpi=300)
            plt.close()

    # 保存每个分区的数据到CSV
    def save_partition_data(self, save_dir="partition_data"):
        """保存每个分区的原始数据到CSV文件"""
        os.makedirs(save_dir, exist_ok=True)
        for leaf, data in self.partition_data.items():
            file_path = os.path.join(save_dir, f"partition_{leaf}_data.csv")
            data.to_csv(file_path, index=False)
            print(f"分区 {leaf} 数据已保存至: {file_path}")

    # 绘制每个分区的预测结果散点图
    def plot_partition_results(self, save_dir="partition_results"):
        """绘制每个分区的预测结果散点图"""
        os.makedirs(save_dir, exist_ok=True)

        for leaf in self.symbolic_models.keys():
            if leaf not in self.partition_data:
                continue

            partition_df = self.partition_data[leaf]
            X_leaf = partition_df[feature_cols]
            y_leaf = partition_df[target_col]

            # 使用该分区的模型进行预测
            y_pred = self.symbolic_models[leaf].predict(X_leaf.values)
            r2 = r2_score(y_leaf, y_pred)
            expression = self.expressions[leaf]

            # 创建图形
            plt.figure(figsize=(10, 8), dpi=120)
            plt.scatter(y_leaf, y_pred, alpha=0.6, color='#1f77b4')

            # 添加对角线
            min_val = min(y_leaf.min(), y_pred.min())
            max_val = max(y_leaf.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            plt.xlabel('实际值', fontsize=12)
            plt.ylabel('预测值', fontsize=12)
            plt.title(f'分区 {leaf} 预测结果', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.3)

            # 添加统计信息和公式
            wrapped_expr = textwrap.fill(expression, width=50)
            plt.text(0.05, 0.95,
                     f'表达式:\n{wrapped_expr}\n\nR² = {r2:.4f}\n样本数 = {len(y_leaf)}',
                     transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(facecolor='white', alpha=0.7))

            # 保存图像
            plt.savefig(f"{save_dir}/partition_{leaf}_scatter.png", bbox_inches='tight', dpi=300)
            plt.close()
            print(f"分区 {leaf} 散点图已保存至: {save_dir}/partition_{leaf}_scatter.png")


# ====================== 参数配置区域 ======================
# 配置决策树分区参数
tree_params = {
    'n_partitions': 5,  # 期望的分区数量
    'max_depth': 5,  # 决策树最大深度
}

# 配置符号回归参数
sr_params = {
    'population_size': 6000,  # 种群大小
    'generations': 50,  # 进化代数
    'tournament_size': 20,  # 锦标赛选择大小
    'stopping_criteria': 0.01,  # 停止标准
    'const_range': (-1.0, 1.0),  # 常数范围
    'init_depth': (2, 6),  # 初始深度范围
    'function_set': ('add', 'sub', 'mul', 'div', 'sqrt', 'log',
                     'abs', 'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan'),
    'parsimony_coefficient': 0.01,  # 简约系数
    'metric': 'rmse'  # 评估指标
}
# ========================================================

# 训练与评估
print("开始训练PS-Tree模型...")
print(f"决策树参数: 分区数={tree_params['n_partitions']}, 最大深度={tree_params['max_depth']}")
ps_tree = PSTree(tree_params=tree_params, sr_params=sr_params)
ps_tree.fit(X_train, y_train)
ps_tree.plot_training_history()  # 生成历史图表

# 保存分区数据和绘制分区结果
ps_tree.save_partition_data()
ps_tree.plot_partition_results()

# 预测与评估
print("\n在测试集上进行评估...")
y_pred = ps_tree.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n评估结果:\nMSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.4f}")

# 输出分区信息
print("\n=== 分区详情 ===")
for i, leaf in enumerate(ps_tree.symbolic_models.keys()):
    print(f"\n分区 {i + 1} (叶节点 {leaf}):")
    print(f"  样本数: {ps_tree.sample_counts[leaf]}")
    print(f"  表达式: {ps_tree.expressions[leaf]}")
    if leaf in ps_tree.histories:
        best_rmse = min(ps_tree.histories[leaf]['best_fitness'])
        print(f"  最佳RMSE: {best_rmse:.6f}")

# 保存模型
print("\n保存模型...")
joblib.dump(ps_tree, 'ps_tree_model.pkl', compress=3)
print("模型已保存为 'ps_tree_model.pkl'")

# 创建包含实际值和预测值的DataFrame
scatter_data = pd.DataFrame({
    'Actual_Values': y_test.values,
    'Predicted_Values': y_pred
})

# 添加索引列以便追踪原始数据位置
scatter_data.reset_index(inplace=True, drop=False)
scatter_data.rename(columns={'index': 'Original_Index'}, inplace=True)

# 保存到CSV文件
scatter_data.to_csv('scatter_plot_data.csv', index=False)
print("散点图数据已保存到 scatter_plot_data.csv")

# 可视化预测结果（高清）
plt.figure(figsize=(10, 8), dpi=120)
plt.scatter(y_test, y_pred, alpha=0.6, color='#1f77b4')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
plt.xlabel('实际值', fontsize=12)
plt.ylabel('预测值', fontsize=12)
plt.title('PS-Tree 预测结果', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.3)

# 添加统计信息
plt.text(0.05, 0.95,
         f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}',
         transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.7))

plt.savefig('ps_tree_prediction.png', bbox_inches='tight', dpi=300)
print("整体预测结果图已保存为 'ps_tree_prediction.png'")
# plt.show()