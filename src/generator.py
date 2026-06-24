from huggingface_hub import InferenceClient
from src.config import HF_TOKEN


class ComplaintGenerator:

    def __init__(self):

        self.client = InferenceClient(
            api_key=HF_TOKEN
        )

        self.model = "meta-llama/Llama-3.1-8B-Instruct"

    def generate(
        self,
        prompt
    ):

        completion = self.client.chat_completion(

            model=self.model,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            max_tokens=300
        )

        return completion.choices[0].message.content