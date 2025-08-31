# src/model_evaluation.py
import pandas as pd, numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, ConfusionMatrixDisplay, roc_auc_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt, seaborn as sns

def eval_metrics(ytrue, ypred, model_name='model'):
    """Calculates and returns key classification metrics."""
    precision, recall, fscore, _ = precision_recall_fscore_support(ytrue, ypred, pos_label=1, average='binary')
    return {
        'model': model_name, 'accuracy': accuracy_score(ytrue, ypred),
        'precision': precision, 'recall': recall, 'fscore': fscore,
        'auc': roc_auc_score(ytrue, ypred)
    }

def visualize_results(results: list):
    """Creates a bar plot to compare model performance."""
    df_results = pd.DataFrame(data=results)
    df_results.set_index('model', inplace=True)
    print(df_results)
    df_results[['precision', 'recall', 'fscore', 'auc']].plot(
        kind='bar', figsize=(15, 8), rot=45, colormap='Paired'
    )
    plt.title('Performance Metrics by Model')
    plt.ylabel('Score')
    plt.show()

def plot_feature_importance(model, X_test, y_test, feature_names):
    """Creates a box plot of permutation feature importance on the test set."""
    print("Analyzing feature importance for the best model...")
    feature_importances = permutation_importance(
        estimator=model, X=X_test, y=y_test, n_repeats=5,
        random_state=123, n_jobs=-1
    )
    sorted_idx = feature_importances.importances_mean.argsort()
    fig, ax = plt.subplots()
    fig.set_figwidth(16)
    fig.set_figheight(10)
    fig.tight_layout()
    ax.boxplot(feature_importances.importances[sorted_idx].T[:10],
               vert=False, tick_labels=feature_names[sorted_idx])
    ax.set_title("Permutation Importances (test set)")
    plt.show()