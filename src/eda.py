import pandas as pd
def dataset_summary(df):
    """
    Display the dataset shape, data types,
    missing values, and unique value counts.
    """

    summary = pd.DataFrame({
        "Data Type": df.dtypes,
        "Missing": df.isnull().sum(),
        "Unique": df.nunique()
    })

    return summary
def product_distribution(df):
    """
    Count complaints for each product.
    """

    return df["Product"].value_counts()

def top_issues(df, top_n=10):
    """
    Return the most common complaint issues.
    """

    return df["Issue"].value_counts().head(top_n)

def narrative_statistics(df):
    """
    Return descriptive statistics for narrative length.
    """

    return df["word_count"].describe()