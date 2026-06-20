import pandas as pd
import re

def missing_values(self):
        missing = self.df.isnull().sum()

        return pd.DataFrame({
            "Missing Count": missing,
            "Missing Percentage":
            round((missing / len(self.df)) * 100, 2)
        }).sort_values(
            "Missing Count",
            ascending=False
        )

def duplicate_count(self):
    return self.df.duplicated().sum()
def remove_duplicates(df):
    """
    Remove duplicate complaint records.
    """

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(f"Removed {before-after} duplicate rows.")

    return df

def remove_empty_narratives(df):
    """
    Remove complaints that do not contain a consumer narrative.

    Only complaints with narratives can be embedded for semantic search.
    """

    before = len(df)

    df = df[
        df["Consumer complaint narrative"]
        .notna()
    ]

    after = len(df)

    print(f"Removed {before-after} complaints without narratives.")

    return df

def clean_narratives(df):
    """
    Clean complaint narratives by removing leading and trailing spaces.
    """

    df["Consumer complaint narrative"] = (
        df["Consumer complaint narrative"]
        .str.strip()
    )

    return df

def add_word_count(df):
    """
    Create a new feature representing
    the number of words in each complaint narrative.
    """

    df["word_count"] = (
        df["Consumer complaint narrative"]
        .fillna("")
        .apply(lambda x: len(x.split()))
    )

    return df

def categorize_products(df):
    """
    Standardize CFPB product names into the four target product categories.
    """

    product_mapping = {
        "Credit card": "Credit Card",
        "Credit card or prepaid card": "Credit Card",

        "Consumer Loan": "Personal Loan",
        "Payday loan": "Personal Loan",
        "Payday loan, title loan, or personal loan": "Personal Loan",
        "Vehicle loan or lease": "Personal Loan",
        "Student loan": "Personal Loan",

        "Checking or savings account": "Savings Account",

        "Money transfers": "Money Transfer",
        "Money transfer, virtual currency, or money service": "Money Transfer",
        "Virtual currency": "Money Transfer"
    }

    df["Product Category"] = df["Product"].map(product_mapping)

    return df

def filter_target_products(df):
    """
    Retain only complaints belonging to the four target products.
    """

    target_products = [
        "Credit Card",
        "Personal Loan",
        "Savings Account",
        "Money Transfer"
    ]

    before = len(df)

    df = df[df["Product Category"].isin(target_products)].copy()

    after = len(df)

    print(f"Removed {before-after:,} complaints from non-target products.")
    print(f"Remaining complaints: {after:,}")

    return df

def clean_text(text):
    """
    Clean consumer complaint narratives for semantic search.

    Steps:
    1. Convert text to lowercase.
    2. Remove common boilerplate phrases.
    3. Remove URLs and email addresses.
    4. Remove special characters.
    5. Normalize whitespace.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Cleaned complaint narrative.
    """

    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove common CFPB boilerplate phrases
    boilerplate_patterns = [
        r"i am writing to file a complaint",
        r"i am writing to complain",
        r"this complaint is about",
        r"i would like to file a complaint",
        r"please help me",
        r"thank you",
    ]

    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def clean_narratives(df):
    """
    Apply text cleaning to all complaint narratives.
    """

    df["Consumer complaint narrative"] = (
        df["Consumer complaint narrative"]
        .apply(clean_text)
    )

    print("Consumer complaint narratives cleaned successfully.")

    return df