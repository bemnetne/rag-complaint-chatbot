import pandas as pd
import chromadb
from tqdm.auto import tqdm

import chromadb


def load_vector_store(
    persist_directory="data/vector_store",
    collection_name="complaints"
):
    """
    Load an existing ChromaDB collection.
    """

    client = chromadb.PersistentClient(
        path=persist_directory
    )

    collection = client.get_collection(
        name=collection_name
    )

    print(
        f"Loaded collection with {collection.count():,} chunks"
    )

    return collection

def build_vector_store(
    parquet_path,
    persist_directory="data/vector_store",
    collection_name="complaints"
):
    """
    Build ChromaDB vector store from the provided parquet file.
    """

    df = pd.read_parquet(parquet_path)

    print(f"Loaded {len(df):,} records")

    client = chromadb.PersistentClient(
        path=persist_directory
    )

    try:
        client.delete_collection(collection_name)
    except:
        pass

    collection = client.get_or_create_collection(
        name=collection_name
    )

    batch_size = client.get_max_batch_size()

    for start in tqdm(
        range(0, len(df), batch_size),
        desc="Indexing Chunks"
    ):

        end = min(start + batch_size, len(df))

        batch = df.iloc[start:end]

        collection.add(
            ids=batch["id"].astype(str).tolist(),
            embeddings=batch["embedding"].tolist(),
            documents=batch["document"].tolist(),
            metadatas=batch["metadata"].tolist()
        )

    print(
        f"Successfully indexed {collection.count():,} chunks"
    )

    return collection