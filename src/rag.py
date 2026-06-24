from src.retriever import ComplaintRetriever
from src.generator import ComplaintGenerator
from src.prompt import create_prompt


class ComplaintRAG:

    def __init__(self):

        self.retriever = ComplaintRetriever(
            persist_directory="../data/vector_store"
        )

        self.generator = ComplaintGenerator()

    def ask(
        self,
        question,
        k=5
    ):

        results = self.retriever.retrieve(
            question,
            k
        )

        context = "\n\n".join(
            results["documents"][0]
        )

        prompt = create_prompt(
            context=context,
            question=question
        )

        answer = self.generator.generate(
            prompt
        )

        return {
            "question": question,
            "answer": answer,
            "sources": results["metadatas"][0],
            "documents": results["documents"][0],
        }