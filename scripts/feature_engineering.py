import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder


# Function to create aggregate features
def create_aggregate_features(df):
    df['TotalTransactionAmount'] = df.groupby('CustomerId')['Amount'].transform('sum')
    df['AvgTransactionAmount'] = df.groupby('CustomerId')['Amount'].transform('mean')
    df['TransactionCount'] = df.groupby('CustomerId')['TransactionId'].transform('count')
    df['StdDevTransactionAmount'] = df.groupby('CustomerId')['Amount'].transform('std')
    return df

# Function to extract temporal features
# Function to extract temporal features
def extract_temporal_features(df):
    # Rename 'TransactionStartTime' to 'TransactionDate' or use 'TransactionStartTime' directly
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year
    return df


# One-Hot Encoding function for categorical variables
def one_hot_encode(df, columns):
    return pd.get_dummies(df, columns=columns, drop_first=True)

# Label Encoding function for categorical variables
def label_encode(df, columns):
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col])
    return df

# Function to handle missing values
def handle_missing_values(df, strategy='median'):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy == 'knn':
                from sklearn.impute import KNNImputer
                imputer = KNNImputer(n_neighbors=5)
                df[col] = imputer.fit_transform(df[[col]])
            else:
                df[col].fillna(df[col].median(), inplace=True)
    return df

# Function to normalize or standardize numerical features
def scale_numerical_features(df, columns, method='normalize'):
    if method == 'normalize':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

# Function to apply WOE and IV using the xverse library

