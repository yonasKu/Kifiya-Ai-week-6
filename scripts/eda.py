
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to display the overview of the dataset
# def data_overview(df):
#     print("Data Overview:")
#     print(df.info())
#     print("\nFirst few rows of the dataset:")
#     print(df.head())
def data_overview(df):
    print("Data Overview:")
    print(df.info())
    print("\nFirst few rows of the dataset:")
    print(df.head())
    print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")

# Function to display summary statistics
# def summary_statistics(df):
#     print("\nSummary Statistics:")
#     print(df.describe())
def summary_statistics(df):
    print("\nSummary Statistics:")
    print(df.describe(include='all'))  # Shows both numerical and categorical summary stats

# Function to identify missing valuesdef missing_values(df):

def missing_values(df):
    print("\nMissing Values:")
    print(df.isnull().sum())
    # Visualizing missing values
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()

# Function to visualize distribution of numerical features

def plot_numerical_distributions(df):
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numerical_columns].hist(figsize=(15, 10), bins=20, color='skyblue', edgecolor='black')
    plt.tight_layout()
    plt.show()

    # KDE Plots (optional)
    for col in numerical_columns:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(df[col], shade=True, color='blue')
        plt.title(f'Distribution of {col} with KDE')
        plt.show()

# Function to visualize distribution of categorical features
def plot_categorical_distributions(df):
    categorical_columns = df.select_dtypes(include=[object, 'category']).columns.tolist()
    for col in categorical_columns:
        if df[col].nunique() > 20:  # Skipping high cardinality columns
            print(f"Skipping {col} due to high cardinality ({df[col].nunique()} unique values)")
            continue
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col, palette='Set2', order=df[col].value_counts().index)
        plt.xticks(rotation=45)
        plt.title(f'Distribution of {col}')
        plt.show()

# Function to plot correlation heatmap
def plot_correlation_heatmap(df):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate the correlation matrix
    correlation_matrix = numeric_df.corr()
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1, linecolor='black',
                mask=np.triu(correlation_matrix))  # Masking the upper triangle for better readability
    plt.title('Correlation Heatmap')
    plt.show()
    
# Function to detect outliers using boxplots
def detect_outliers(df):
    # Select only numeric columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numerical_columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x=col, color='lightcoral')
        plt.title(f'Boxplot of {col}')
        plt.show()

        # Detecting outliers using IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
        print(f"\nOutliers detected in {col}: {len(outliers)}")


# Example usage of optional cleaning functions:
# Fill missing values with the median (example)
def fill_missing_values(df):
    for col in df.columns:
        if df[col].dtype == 'object':  # For categorical columns
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:  # For numerical columns
            df[col].fillna(df[col].median(), inplace=True)
    return df
