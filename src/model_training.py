# src/model_training.py
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier

def train_model_with_grid_search(model, X_train, y_train, param_grid):
    """
    Trains a model using GridSearchCV for hyperparameter tuning.
    
    Returns the best_estimator_ from the grid search.
    """
    search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=3, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    return search.best_estimator_

def train_all_models(X_train_scaled, y_train, X_train):
    """
    Trains all individual models using the appropriate data splits.
    
    Returns a dictionary of trained models.
    """
    models = {}
    print("Training Logistic Regression...")
    models['logreg'] = train_model_with_grid_search(
        LogisticRegression(random_state=666), X_train_scaled, y_train, {'class_weight': [{0: 0.05, 1: 0.95}]}
    )
    print("Training K-Nearest Neighbors...")
    models['knn'] = train_model_with_grid_search(
        KNeighborsClassifier(), X_train_scaled, y_train, {'n_neighbors': [3, 5, 7]}
    )
    print("Training Support Vector Classifier...")
    models['svc'] = train_model_with_grid_search(
        SVC(random_state=666), X_train_scaled, y_train, {'C': [1, 10], 'kernel': ['rbf']}
    )
    print("Training Bagging Classifier...")
    models['bagging'] = train_model_with_grid_search(
        BaggingClassifier(DecisionTreeClassifier(random_state=666)), X_train, y_train, {'n_estimators': [20, 30, 40, 50]}
    )
    print("Training Random Forest...")
    models['rforest'] = train_model_with_grid_search(
        RandomForestClassifier(random_state=666), X_train, y_train, {'n_estimators': [20, 30, 40, 50], 'max_depth': [10, 20, 30]}
    )
    print("Training XGBoost...")
    models['xgboost'] = train_model_with_grid_search(
        XGBClassifier(objective='binary:logistic', eval_metric='logloss'), X_train, y_train,
        {'learning_rate': [0.1, 0.2, 0.3, 0.4], 'n_estimators': [10, 20, 30, 40, 50]}
    )
    return models