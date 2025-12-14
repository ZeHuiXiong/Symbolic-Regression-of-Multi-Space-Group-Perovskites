import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import tqdm
import os
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免在服务器上出现问题
import matplotlib.pyplot as plt
from IPython.display import clear_output  # 用于在Jupyter中清除输出


class SymbolicRegressionTree:
    def __init__(self, global_precision=0.01, local_precision=0.005,
                 data_ratio=0.7, learning_rate=0.1,
                 weight_learning_rate=0.01, max_iter=100, min_improvement=1e-5,
                 clip_value=10.0, range_extension=0.2, smoothness=0.1,
                 live_display=False, display_interval=10, output_dir="live_display"):
        """
        Initialize the Symbolic Regression Tree model

        Parameters:
        global_precision: Precision for global distribution sampling
        local_precision: Precision for local feature value sampling
        data_ratio: Proportion of training data used in each iteration
        learning_rate: Learning rate for function value updates
        weight_learning_rate: Learning rate for weight updates
        max_iter: Maximum number of iterations
        min_improvement: Minimum improvement threshold for early stopping
        clip_value: Gradient clipping threshold
        range_extension: Feature value range extension ratio (0-1)
        smoothness: Second derivative continuity constraint strength (0-1)
        live_display: Whether to display live training progress
        display_interval: How often to update the display (in iterations)
        output_dir: Directory to save live display images
        """
        self.global_precision = global_precision
        self.local_precision = local_precision
        self.data_ratio = data_ratio
        self.learning_rate = learning_rate
        self.weight_learning_rate = weight_learning_rate
        self.max_iter = max_iter
        self.min_improvement = min_improvement
        self.clip_value = clip_value
        self.range_extension = range_extension
        self.smoothness = smoothness
        self.feature_functions = {}
        self.feature_weights = None
        self.history = {'train_mse': [], 'test_mse': [], 'r2': []}
        self.feature_names = None
        self.live_display = live_display
        self.display_interval = display_interval
        self.output_dir = output_dir

        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _initialize_feature_functions(self, X, y):
        """Initialize piecewise interpolation functions for each feature"""
        self.feature_functions = {}
        n_features = X.shape[1]
        global_mean = np.mean(y)

        # Initialize weights - equal for all features
        self.feature_weights = np.ones(n_features) / n_features

        for i, feature_name in enumerate(self.feature_names):
            feature_values = X[:, i]
            min_value = np.min(feature_values)
            max_value = np.max(feature_values)

            # Calculate extended range
            ext_min = min_value * (1 - self.range_extension)
            ext_max = max_value * (1 + self.range_extension)

            # Uniform sampling in global distribution (global precision)
            global_points = np.arange(ext_min, ext_max, self.global_precision)

            # Dense sampling in feature value fluctuation range (local precision)
            local_points = []
            for value in feature_values:
                local_min = max(ext_min, value - self.local_precision * 5)
                local_max = min(ext_max, value + self.local_precision * 5)

                if local_min < local_max:
                    num_points = max(2, int((local_max - local_min) / self.local_precision) + 1)
                    points = np.linspace(local_min, local_max, num_points)
                    local_points.extend(points)

            # Combine all points and sort
            all_points = np.unique(np.concatenate([global_points, np.array(local_points)]))
            all_points.sort()

            # Initialize function values
            function_values = np.full(len(all_points), global_mean * self.feature_weights[i])

            # Apply smoothness constraint
            if self.smoothness > 0:
                function_values = self._apply_smoothness_constraint(all_points, function_values)

            # Create interpolation function
            self.feature_functions[feature_name] = {
                'points': all_points,
                'values': function_values,
                'function': interp1d(all_points, function_values,
                                     kind='linear', bounds_error=False,
                                     fill_value=(function_values[0], function_values[-1]))
            }

    def _apply_smoothness_constraint(self, points, values):
        """Apply second derivative continuity constraint"""
        if len(points) < 4 or self.smoothness <= 0:
            return values

        spline = UnivariateSpline(points, values, s=self.smoothness * len(points))
        return spline(points)

    def predict(self, X):
        """Make predictions using current function set"""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        for i, feature_name in enumerate(self.feature_names):
            feature_values = X[:, i]
            f = self.feature_functions[feature_name]['function']
            predictions += self.feature_weights[i] * f(feature_values)

        return predictions

    def _update_feature_functions(self, X_batch, residuals):
        """Update feature functions and weights based on residuals"""
        n_features = X_batch.shape[1]
        n_samples = X_batch.shape[0]

        # Calculate feature contributions
        feature_contributions = np.zeros((n_samples, n_features))
        for i, feature_name in enumerate(self.feature_names):
            feature_values = X_batch[:, i]
            f = self.feature_functions[feature_name]['function']
            feature_contributions[:, i] = f(feature_values)

        # Update weights
        for i in range(n_features):
            cov = np.cov(feature_contributions[:, i], residuals)[0, 1]
            cov = np.clip(cov, -self.clip_value, self.clip_value)
            self.feature_weights[i] += self.weight_learning_rate * cov

        # Normalize weights
        self.feature_weights = np.abs(self.feature_weights)
        self.feature_weights /= np.sum(self.feature_weights)

        # Update function values
        for i, feature_name in enumerate(self.feature_names):
            feature_values = X_batch[:, i]
            f_info = self.feature_functions[feature_name]

            for j, point in enumerate(f_info['points']):
                distances = np.abs(feature_values - point)
                nearby_indices = np.where(distances < self.local_precision * 5)[0]

                if len(nearby_indices) > 0:
                    avg_residual = np.mean(residuals[nearby_indices])
                    avg_residual = np.clip(avg_residual, -self.clip_value, self.clip_value)
                    update = self.learning_rate * avg_residual / (self.feature_weights[i] + 1e-7)
                    f_info['values'][j] += update

            # Apply smoothness constraint
            if self.smoothness > 0:
                f_info['values'] = self._apply_smoothness_constraint(f_info['points'], f_info['values'])

            # Update interpolation function
            f_info['function'] = interp1d(f_info['points'], f_info['values'],
                                          kind='linear', bounds_error=False,
                                          fill_value=(f_info['values'][0], f_info['values'][-1]))

    def fit(self, X, y, feature_names, test_size=0.2, random_state=42):
        """Train the symbolic regression model with live display"""
        self.feature_names = feature_names

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Initialize feature functions
        self._initialize_feature_functions(X_train, y_train)

        # Initial predictions
        train_pred = self.predict(X_train)
        test_pred = self.predict(X_test)

        # Handle NaN predictions
        if np.isnan(train_pred).any() or np.isnan(test_pred).any():
            self._initialize_feature_functions(X_train, y_train)
            train_pred = self.predict(X_train)
            test_pred = self.predict(X_test)

        # Record initial performance
        best_train_mse = mean_squared_error(y_train, train_pred)
        best_test_mse = mean_squared_error(y_test, test_pred)
        best_r2 = r2_score(y_test, test_pred)

        self.history['train_mse'].append(best_train_mse)
        self.history['test_mse'].append(best_test_mse)
        self.history['r2'].append(best_r2)

        # Create progress bar
        progress_bar = tqdm(total=self.max_iter, desc="Training Progress")

        # Training loop
        for iter in range(1, self.max_iter + 1):
            # Create batch
            n_samples = X_train.shape[0]
            batch_size = int(n_samples * self.data_ratio)
            batch_indices = np.random.choice(n_samples, batch_size, replace=False)
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Predict and calculate residuals
            batch_pred = self.predict(X_batch)
            if np.isnan(batch_pred).any():
                progress_bar.update(1)
                continue
            residuals = y_batch - batch_pred

            # Update functions
            self._update_feature_functions(X_batch, residuals)

            # Evaluate model
            train_pred = self.predict(X_train)
            test_pred = self.predict(X_test)

            if np.isnan(train_pred).any() or np.isnan(test_pred).any():
                progress_bar.update(1)
                continue

            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            r2 = r2_score(y_test, test_pred)

            self.history['train_mse'].append(train_mse)
            self.history['test_mse'].append(test_mse)
            self.history['r2'].append(r2)

            # Update progress bar
            progress_bar.set_postfix({
                'Train MSE': f"{train_mse:.6f}",
                'Test MSE': f"{test_mse:.6f}",
                'R²': f"{r2:.4f}"
            })
            progress_bar.update(1)

            # Live display
            if self.live_display and iter % self.display_interval == 0:
                self.update_live_display(iter, X_test, y_test)

            # Check for improvement
            improvement = self.history['test_mse'][-2] - test_mse if len(self.history['test_mse']) > 1 else 0

            # Early stopping
            if abs(improvement) < self.min_improvement and iter > 10:
                progress_bar.close()
                print(f"Stopping early at iteration {iter} - improvement below threshold")
                break

        progress_bar.close()

    def update_live_display(self, iteration, X_test, y_test):
        """Update live display with current training status"""
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Iteration {iteration}', fontsize=16)

        # 1. 性能指标图表
        ax1 = axes[0, 0]
        ax1.plot(self.history['train_mse'], label='Train MSE')
        ax1.plot(self.history['test_mse'], label='Test MSE')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('MSE')
        ax1.set_title('Training and Test MSE')
        ax1.legend()
        ax1.grid(True)

        ax2 = axes[0, 1]
        ax2.plot(self.history['r2'], label='Test R²', color='green')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('R² Score')
        ax2.set_title('Test R² Score')
        ax2.legend()
        ax2.grid(True)

        # 2. 预测值与实际值对比
        ax3 = axes[1, 0]
        predictions = self.predict(X_test)
        if np.isnan(predictions).any():
            predictions = np.zeros_like(predictions)

        min_val = min(np.min(y_test), np.min(predictions))
        max_val = max(np.max(y_test), np.max(predictions))

        ax3.scatter(y_test, predictions, alpha=0.6)
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax3.set_xlabel('Actual Values')
        ax3.set_ylabel('Predicted Values')
        ax3.set_title('Test Set Predictions')
        ax3.grid(True)

        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        ax3.text(0.05, 0.95, f'R² = {r2:.4f}\nMSE = {mse:.6f}',
                 transform=ax3.transAxes, verticalalignment='top')

        # 3. 特征函数图
        ax4 = axes[1, 1]
        n_features = len(self.feature_names)
        for i, feature_name in enumerate(self.feature_names):
            f_info = self.feature_functions[feature_name]
            x_min, x_max = np.min(f_info['points']), np.max(f_info['points'])
            x_dense = np.linspace(x_min, x_max, 500)
            y_dense = f_info['function'](x_dense)

            # 归一化以便在同一图表中显示
            y_dense = (y_dense - np.min(y_dense)) / (np.max(y_dense) - np.min(y_dense) + 1e-7) + i
            ax4.plot(x_dense, y_dense, label=f'{feature_name} (w={self.feature_weights[i]:.4f})')

        ax4.set_title('Feature Functions (Normalized)')
        ax4.set_xlabel('Feature Value')
        ax4.set_ylabel('Normalized Contribution')
        ax4.legend()
        ax4.grid(True)

        # 调整布局并保存
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为标题留出空间
        plt.savefig(os.path.join(self.output_dir, f'live_display_{iteration}.png'))
        plt.close(fig)

        # 在Jupyter环境中清除之前的输出并显示新图像
        try:
            clear_output(wait=True)
            from IPython.display import display, Image
            display(Image(filename=os.path.join(self.output_dir, f'live_display_{iteration}.png')))
        except:
            # 不在Jupyter环境中，只需保存图像
            pass

    def plot_functions(self):
        """Plot interpolation functions for each feature"""
        n_features = len(self.feature_names)
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 3 * n_features))

        if n_features == 1:
            axes = [axes]

        for i, feature_name in enumerate(self.feature_names):
            f_info = self.feature_functions[feature_name]
            ax = axes[i]

            x_min, x_max = np.min(f_info['points']), np.max(f_info['points'])
            x_dense = np.linspace(x_min, x_max, 500)
            y_dense = f_info['function'](x_dense)

            ax.plot(x_dense, y_dense, 'b-', label='Interpolation')
            ax.plot(f_info['points'], f_info['values'], 'ro', markersize=4, label='Control Points')
            ax.set_title(f'Feature: {feature_name} (Weight: {self.feature_weights[i]:.4f})')
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Contribution to Target')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        return fig

    def plot_performance(self):
        """Plot training performance history"""
        fig = plt.figure(figsize=(12, 10))

        plt.subplot(2, 1, 1)
        plt.plot(self.history['train_mse'], label='Train MSE')
        plt.plot(self.history['test_mse'], label='Test MSE')
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.title('Training and Test MSE over Iterations')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(self.history['r2'], label='Test R²', color='green')
        plt.xlabel('Iteration')
        plt.ylabel('R² Score')
        plt.title('Test R² Score over Iterations')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        return fig

    def plot_predictions(self, X, y, title='Predictions vs Actual'):
        """Plot predicted vs actual values"""
        predictions = self.predict(X)
        if np.isnan(predictions).any():
            predictions = np.zeros_like(predictions)

        fig = plt.figure(figsize=(10, 8))
        plt.scatter(y, predictions, alpha=0.6)

        min_val = min(np.min(y), np.min(predictions))
        max_val = max(np.max(y), np.max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.grid(True)

        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}\nMSE = {mse:.6f}',
                 transform=plt.gca().transAxes, verticalalignment='top')

        return fig

    def get_feature_functions(self):
        return self.feature_functions

    def get_feature_weights(self):
        return self.feature_weights