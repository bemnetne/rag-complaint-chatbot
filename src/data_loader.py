import pandas as pd


def load_data(filepath):
    """
    Load CFPB complaint dataset.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    DataFrame
        Loaded complaint dataset.
    """

    df = pd.read_csv(filepath)

    print(f"Dataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df