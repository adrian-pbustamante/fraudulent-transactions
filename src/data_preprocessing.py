# src/data_preprocessing.py
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(dataset_name="qnqfbqfqo/credit-card-fraud-detection-date-25th-of-june-2015"):
    """Downloads and loads the dataset from KaggleHub."""
    path = kagglehub.dataset_download(dataset_name)
    data = pd.read_csv(f"{path}/creditcard_csv.csv")
    return data

def preprocess_data(df: pd.DataFrame):
    """Performs data cleaning and feature engineering."""
    df.drop('Time', axis=1, inplace=True)
    df['Class'] = df['Class'].apply(lambda x: int(x[1]))
    df.drop_duplicates(keep='first', inplace=True)
    return df

def create_train_test_splits(data: pd.DataFrame, test_size=0.5):
    """Creates stratified train-test splits with raw data."""
    X = data.drop('Class', axis=1).copy()
    y = data['Class'].astype(int).copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=666
    )
    return X_train, X_test, y_train, y_test