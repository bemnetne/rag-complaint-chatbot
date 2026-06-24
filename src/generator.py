from huggingface_hub import InferenceClient
import os
from src.config import HF_TOKEN
class ComplaintGenerator:

    def __init__(self):

        self.client = InferenceClient(
            token=HF_TOKEN
        )

        self.model = "Qwen/Qwen2.5-7B-Instruct"

    def generate(self, prompt):

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