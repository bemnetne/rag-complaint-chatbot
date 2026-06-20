import matplotlib.pyplot as plt
import seaborn as sns

def plot_product_distribution(df):
    """
    Visualize complaint counts across financial products.
    """

    plt.figure(figsize=(10,6))

    sns.countplot(
        y="Product",
        data=df,
        order=df["Product"].value_counts().index
    )

    plt.title("Complaint Distribution Across Products")
    plt.xlabel("Number of Complaints")
    plt.ylabel("Product")

    plt.show()

def plot_narrative_length(df):
    """
    Plot the distribution of complaint narrative lengths.
    """

    plt.figure(figsize=(10,5))

    sns.histplot(
        df["word_count"],
        bins=50,
        kde=True
    )

    plt.title("Narrative Length Distribution")

    plt.xlabel("Word Count")

    plt.show()

def plot_narrative_availability(df):
    """
    Visualize complaints with and without narratives.
    """

    counts = [
        df["Consumer complaint narrative"].notna().sum(),
        df["Consumer complaint narrative"].isna().sum()
    ]

    plt.figure(figsize=(6,6))

    plt.pie(
        counts,
        labels=["With Narrative","Without Narrative"],
        autopct="%1.1f%%"
    )

    plt.title("Availability of Complaint Narratives")

    plt.show()

def plot_top_complaint_issues(df, top_n=10):
    """
    Plot the top N most frequent complaint issues.

    Parameters
    ----------
    df : pandas.DataFrame
        CFPB complaints dataset.
    top_n : int, default=10
        Number of top complaint issues to display.
    """

    # Count complaint issues
    issue_counts = (
        df["Issue"]
        .value_counts()
        .head(top_n)
        .sort_values()
    )

    # Create figure
    plt.figure(figsize=(12, 6))

    sns.barplot(
        x=issue_counts.values,
        y=issue_counts.index,
        palette="Blues_r"
    )

    plt.title(f"Top {top_n} Complaint Issues", fontsize=15, fontweight="bold")
    plt.xlabel("Number of Complaints", fontsize=12)
    plt.ylabel("Complaint Issue", fontsize=12)

    # Format x-axis with commas
    plt.gca().xaxis.set_major_formatter(
        plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}')
    )

    # Add value labels
    for i, value in enumerate(issue_counts.values):
        plt.text(
            value,
            i,
            f" {value:,}",
            va="center",
            fontsize=10
        )

    plt.tight_layout()
    plt.show()