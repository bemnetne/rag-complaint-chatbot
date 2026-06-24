from sentence_transformers import SentenceTransformer
import chromadb


class ComplaintRetriever:

    def __init__(
        self,
        persist_directory="../data/vector_store",
        collection_name="complaints"
    ):

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = client.get_collection(
            collection_name
        )

    def retrieve(
        self,
        question,
        k=5
    ):
        """
        Retrieve top-k relevant complaint chunks.
        """

        query_embedding = self.embedding_model.encode(
            question,
            convert_to_numpy=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        return results
    def get_context(
        self,
        question,
        k=5
    ):

        results = self.retrieve(
            question,
            k=k
        )

        context = "\n\n".join(
            results["documents"][0]
        )

        return context, results