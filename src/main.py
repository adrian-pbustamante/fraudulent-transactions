# src/main.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_preprocessing import load_data, preprocess_data, create_train_test_splits
from model_training import train_all_models
from model_evaluation import eval_metrics, visualize_results, plot_feature_importance


def main():
    """Main function to run the fraud detection pipeline."""
    #  Data Preparation
    print("Loading and preprocessing data...")
    df = load_data()
    df_clean = preprocess_data(df)

    #  Create Data Splits (Raw Data)
    print("Creating raw data splits...")
    X_train, X_test, y_train, y_test = create_train_test_splits(df_clean)

    #  Scaling the Data (Correctly)
    print("Scaling data (fitting on training set only)...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #  Model Training
    print("Training all models...")
    models = train_all_models(X_train_scaled, y_train, X_train)

    # Model Evaluation
    print(" Evaluating models and plotting results...")
    results = []
    feature_names = df_clean.drop('Class', axis=1).columns

    for name, model in models.items():
        if name in ['bagging', 'rforest', 'xgboost']:
            X_test_eval = X_test
        else:
            X_test_eval = X_test_scaled
        
        y_pred = model.predict(X_test_eval)
        result = eval_metrics(y_test, y_pred, model_name=name)
        results.append(result)

    visualize_results(results)

    # Model Interpretability
    best_model_name = 'xgboost'
    best_model = models[best_model_name]
    #breakpoint()
    plot_feature_importance(best_model, X_test, y_test, feature_names)

if __name__ == "__main__":
    main()