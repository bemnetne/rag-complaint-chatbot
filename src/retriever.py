from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path



class ComplaintRetriever:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    def __init__(
        self,
        persist_directory=(PROJECT_ROOT/"data"/"vector_store"),
        collection_name="complaints"
    ):

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        chroma_client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = chroma_client.get_collection(
            collection_name
        )

    def retrieve(self, question, k=5):

        query_embedding = self.embedding_model.encode(
            question,
            convert_to_numpy=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        return results