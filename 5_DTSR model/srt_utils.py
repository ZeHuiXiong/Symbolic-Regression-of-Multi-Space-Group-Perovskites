import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt


def ensure_directory_exists(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_plot(fig, filename, output_dir="output"):
    """Save matplotlib figure to file"""
    output_path = ensure_directory_exists(output_dir)
    filepath = os.path.join(output_path, filename)
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    return filepath


def save_prediction_data(X, y, predictions, feature_names, filename, output_dir="output"):
    """Save prediction results to CSV"""
    output_path = ensure_directory_exists(output_dir)
    filepath = os.path.join(output_path, filename)

    data = pd.DataFrame({
        'Actual': y,
        'Predicted': predictions
    })

    for i, col in enumerate(feature_names):
        data[col] = X[:, i]

    data.to_csv(filepath, index=False)
    return filepath


def save_model(model, filename, output_dir="output"):
    """Save model to file"""
    output_path = ensure_directory_exists(output_dir)
    filepath = os.path.join(output_path, filename)
    joblib.dump(model, filepath)
    return filepath


def save_weights(weights, feature_names, filename, output_dir="output"):
    """Save feature weights to CSV"""
    output_path = ensure_directory_exists(output_dir)
    filepath = os.path.join(output_path, filename)

    data = pd.DataFrame({
        'Feature': feature_names,
        'Weight': weights
    })
    data.to_csv(filepath, index=False)
    return filepath


def save_feature_functions(feature_functions, output_dir="output"):
    """Save feature function data to CSV files"""
    output_path = ensure_directory_exists(output_dir)

    for feature, func_info in feature_functions.items():
        func_data = pd.DataFrame({
            'feature_value': func_info['points'],
            'contribution': func_info['values']
        })
        filepath = os.path.join(output_path, f"{feature}_function.csv")
        func_data.to_csv(filepath, index=False)

    return output_path