import numpy as np
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load the sentence transformer embedding model.

    Parameters
    ----------
    model_name : str
        Name of the embedding model.

    Returns
    -------
    SentenceTransformer
        Loaded embedding model.
    """

    model = SentenceTransformer(model_name)

    print(f"Loaded embedding model: {model_name}")

    return model

def generate_embeddings(
    df,
    model,
    text_column="chunk",
    batch_size=64
):
    """
    Generate embeddings for each text chunk.

    Parameters
    ----------
    df : pandas.DataFrame
        Chunked complaint dataset.

    model : SentenceTransformer
        Loaded embedding model.

    text_column : str
        Column containing text chunks.

    batch_size : int
        Batch size for encoding.

    Returns
    -------
    pandas.DataFrame
        DataFrame with embeddings.
    """

    embeddings = model.encode(
        df[text_column].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    df = df.copy()
    df["embedding"] = embeddings.tolist()

    return df