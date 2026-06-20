import chromadb
from tqdm.auto import tqdm


def create_chroma_collection(
    collection_name="complaints",
    persist_directory="../vector_store"
):
    """
    Create or load a persistent ChromaDB collection.
    """

    client = chromadb.PersistentClient(path=persist_directory)

    collection = client.get_or_create_collection(
        name=collection_name
    )

    return collection


def add_embeddings(
    collection,
    df,
    embedding_column="embedding",
    text_column="chunk",
    batch_size=None
):
    """
    Store embeddings and metadata in ChromaDB.

    Parameters
    ----------
    collection : chromadb.Collection
    df : pandas.DataFrame
    embedding_column : str
    text_column : str
    batch_size : int, optional
        Maximum number of records per batch.
    """

    # Use ChromaDB's maximum supported batch size
    if batch_size is None:
        batch_size = collection._client.get_max_batch_size()

    required_columns = [
        "Complaint ID",
        "Product Category",
        "Product",
        "Issue",
        "Company",
        "chunk_index",
        "total_chunks",
        text_column,
        embedding_column
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for start in tqdm(range(0, len(df), batch_size), desc="Indexing Chunks"):

        end = min(start + batch_size, len(df))

        batch = df.iloc[start:end]

        collection.add(

            ids=[
                    f"{row['Complaint ID']}_{row['chunk_index']}"
                    for row in batch.to_dict("records")
                ],

            embeddings=batch[embedding_column].tolist(),

            documents=batch[text_column].tolist(),

            metadatas=[
            {
                "complaint_id": str(row["Complaint ID"]),
                "product_category": row["Product Category"],
                "product": row["Product"],
                "issue": row["Issue"],
                "sub_issue": row["Sub-issue"],
                "company": row["Company"],
                "state": row["State"],
                "date_received": str(row["Date received"]),
                "chunk_index": int(row["chunk_index"]),
                "total_chunks": int(row["total_chunks"])
            }

            for row in batch.to_dict("records")
        ]
        )

    print(f"\nSuccessfully indexed {len(df):,} chunks.")

    return collection