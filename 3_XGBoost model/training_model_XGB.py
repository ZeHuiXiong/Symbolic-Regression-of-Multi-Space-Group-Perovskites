
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
from shap import Explanation

# Create output directory
output_dir = "train_out"
os.makedirs(output_dir, exist_ok=True)

# Configure plotting style
sns.set(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.dpi"] = 100

# 1. Data Loading and Preprocessing
file_path = r'./Dateset/7176_key6.csv'
target_column = 'band_gap'
drop_columns = ['stability', 'delta_e', 'struct_type', 'ntypes', 'natoms', 'A_site', 'B_site', 'X_site']
id_columns = ['name', 'spacegroup', 'structure_id']

print("Loading data...")
try:
    data = pd.read_csv(file_path)
    data['structure_id'] = data.index
    print(f"Original data shape: {data.shape}")
except Exception as e:
    print(f"Data loading failed: {e}")
    exit(1)

# Clean data and handle missing values
data_cleaned = data.drop(columns=drop_columns, errors='ignore')
print(f"Cleaned data shape: {data_cleaned.shape}")

print("\nMissing value statistics:")
print(data_cleaned.isnull().sum()[data_cleaned.isnull().sum() > 0])

# Handle missing values
if data_cleaned[target_column].isnull().sum() > 0:
    print(f"Removing {data_cleaned[target_column].isnull().sum()} samples with missing target values")
    data_cleaned = data_cleaned.dropna(subset=[target_column])

for col in data_cleaned.columns:
    if col != target_column and data_cleaned[col].isnull().sum() > 0:
        median_val = data_cleaned[col].median()
        data_cleaned[col] = data_cleaned[col].fillna(median_val)
        print(f"Column '{col}' filled with median: {median_val:.4f}")

# Prepare features and target
X = data_cleaned.drop(columns=[target_column] + id_columns)
y = data_cleaned[target_column]
identifiers = data_cleaned[id_columns]
feature_names = X.columns.tolist()

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Target stats - Train: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
print(f"Target stats - Test: mean={y_test.mean():.4f}, std={y_test.std():.4f}")

# Get test set identifiers
test_identifiers = identifiers.loc[X_test.index].copy()

# 3. Create DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

# 4. XGBoost parameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': ['rmse', 'mae'],
    'booster': 'gbtree',
    'eta': 0.005,
    'max_depth': 9,
    'min_child_weight': 1,
    'subsample': 0.5,
    'colsample_bytree': 1,
    'seed': 2025,
    'verbosity': 1,
}

# 5. Train model with early stopping
print("\nTraining XGBoost model...")
model = xgb.train(
    params,
    dtrain,
    num_boost_round=50000,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=50,
    verbose_eval=50
)

print(f"Best iteration: {model.best_iteration}")
print(f"Best validation score: {model.best_score:.4f}")

# 6. Model evaluation
y_pred = model.predict(dtest)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel performance:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# 7. Save prediction results
results_df = test_identifiers.copy()
results_df['actual_band_gap'] = y_test.values
results_df['predicted_band_gap'] = y_pred
results_df['absolute_error'] = np.abs(y_test.values - y_pred)

results_filename = os.path.join(output_dir, 'test_predictions_results.csv')
results_df.to_csv(results_filename, index=False)
print(f"\nPredictions saved to '{results_filename}'")

# 8. Feature importance analysis
print("\nCalculating feature importance...")
importance = model.get_score(importance_type='gain')
importance_df = pd.DataFrame({
    'Feature': importance.keys(),
    'Importance': importance.values()
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print("\nTop 10 features:")
print(importance_df.head(10))

print("\nStarting SHAP analysis...")
shap.initjs()
explainer = shap.TreeExplainer(model)

# Compute SHAP values
sample_indices = np.random.choice(X_test.index, size=min(1000, len(X_test)), replace=False)
X_test_sample = X_test.loc[sample_indices]
shap_values = explainer.shap_values(X_test_sample)

# Generate SHAP plots
shap_plots = [
    ('dot', "SHAP Feature Importance", 'shap_feature_importance.png'),
    ('bar', "SHAP Feature Importance (Absolute)", 'shap_absolute_importance.png'),
    ('violin', "Feature Impact on Predictions", 'shap_beeswarm_plot.png')
]

for plot_type, title, filename in shap_plots:
    plt.figure()
    shap.summary_plot(shap_values, X_test_sample, plot_type=plot_type, max_display=20, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"Saved {os.path.join(output_dir, filename)}")

# SHAP dependence plots for top features
for feature in importance_df.head(5)['Feature']:
    try:
        plt.figure()
        shap.dependence_plot(feature, shap_values, X_test_sample, show=False)
        plt.title(f"SHAP Dependence Plot for {feature}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_dependence_{feature}.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating dependence plot for {feature}: {e}")

# Individual sample explanations
for i in np.random.choice(range(len(X_test_sample)), 3, replace=False):
    try:
        plt.figure()
        shap.plots.waterfall(
            Explanation(
                values=shap_values[i],
                base_values=explainer.expected_value,
                data=X_test_sample.iloc[i],
                feature_names=X_test_sample.columns
            ),
            max_display=10,
            show=False
        )
        plt.title(f"Waterfall Plot for Sample {i}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_waterfall_sample_{i}.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating explanation for sample {i}: {e}")

# SHAP heatmap
try:
    plt.figure()
    shap.plots.heatmap(
        Explanation(
            values=shap_values[:50],
            base_values=explainer.expected_value,
            data=X_test_sample.iloc[:50],
            feature_names=X_test_sample.columns
        ),
        show=False
    )
    plt.title("SHAP Values Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_heatmap.png'), dpi=150)
    plt.close()
    print(f"Saved {os.path.join(output_dir, 'shap_heatmap.png')}")
except Exception as e:
    print(f"Error creating heatmap: {e}")

print("\nSHAP analysis completed!")

# 9. Visualizations
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

# Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.6, s=30)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
axes[0].plot(lims, lims, 'r--', lw=2)
axes[0].set_title(f'Actual vs Predicted Band Gap (R²={r2:.3f})')
axes[0].set_xlabel('Actual Band Gap')
axes[0].set_ylabel('Predicted Band Gap')
axes[0].grid(True, linestyle='--', alpha=0.3)

# Feature importance
top_features = importance_df.head(20)
axes[1].barh(top_features['Feature'], top_features['Importance'], color='dodgerblue')
axes[1].set_title('Top 20 Feature Importances (Gain)')
axes[1].set_xlabel('Importance Score')
axes[1].invert_yaxis()

# Error distribution
sns.histplot(y_test - y_pred, kde=True, bins=30, ax=axes[2])
axes[2].axvline(x=0, color='r', linestyle='--', lw=2)
axes[2].set_title('Prediction Error Distribution')
axes[2].set_xlabel('Prediction Error')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'xgboost_model_results.png'), dpi=150)
print(f"\nVisualizations saved as '{os.path.join(output_dir, 'xgboost_model_results.png')}'")

# 10. Save model and feature importance
model.save_model(os.path.join(output_dir, 'band_gap_xgboost_model.json'))
importance_df.to_csv(os.path.join(output_dir, 'xgboost_feature_importance.csv'), index=False)
print("Model and feature importance data saved")

print("\nAnalysis completed!")