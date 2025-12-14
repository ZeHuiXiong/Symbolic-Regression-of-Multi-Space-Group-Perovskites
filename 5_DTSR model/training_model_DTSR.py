from sklearn.model_selection import train_test_split
import time
from symbolic_regression_tree import SymbolicRegressionTree
from srt_utils import *


def train_symbolic_regression_model(data_path, feature_cols, target_col, output_dir, model_params):
    """Train symbolic regression model and save results"""
    # Create output directory
    ensure_directory_exists(output_dir)

    # Load data
    data = pd.read_csv(data_path)
    X = data[feature_cols].values
    y = data[target_col].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train model
    model = SymbolicRegressionTree(**model_params)

    start_time = time.time()
    model.fit(X_train, y_train, feature_names=feature_cols)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Save model
    save_model(model, "symbolic_regression_model.pkl", output_dir)

    # Save visualizations
    fig = model.plot_functions()
    save_plot(fig, "feature_functions.png", output_dir)

    fig = model.plot_performance()
    save_plot(fig, "training_performance.png", output_dir)

    fig = model.plot_predictions(X_train, y_train, title='Training Set Predictions')
    save_plot(fig, "training_set_predictions.png", output_dir)

    fig = model.plot_predictions(X_test, y_test, title='Test Set Predictions')
    save_plot(fig, "test_set_predictions.png", output_dir)

    # Save prediction data
    test_pred = model.predict(X_test)
    save_prediction_data(X_test, y_test, test_pred, feature_cols,
                         "test_set_predictions.csv", output_dir)

    train_pred = model.predict(X_train)
    save_prediction_data(X_train, y_train, train_pred, feature_cols,
                         "training_set_predictions.csv", output_dir)

    # Save feature functions and weights
    save_feature_functions(model.get_feature_functions(), output_dir)
    save_weights(model.get_feature_weights(), feature_cols, "feature_weights.csv", output_dir)

    print(f"All outputs saved to '{output_dir}' directory")


if __name__ == "__main__":
    # Configuration
    config = {
        "data_path": r"./Dateset/7176_key6.csv",
        "feature_cols": ['EG_B2', 'EG_B1', 'EG_max', 'r_B2X', 'Nd_max', 'mean_Nd'],
        "target_col": 'band_gap',
        "output_dir": "srt_results",
        "model_params": {
            "global_precision": 0.002,
            "local_precision": 0.001,
            "data_ratio": 0.7,
            "learning_rate": 0.005,
            "weight_learning_rate": 0.005,
            "max_iter": 10000,
            "min_improvement": 1e-5,
            "clip_value": 5.0,
            "range_extension": 0.2,
            "smoothness": 0.00005,
        }
    }

    # Train model
    train_symbolic_regression_model(**config)