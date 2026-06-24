from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError(
        "HF_TOKEN not found. "
        "Please add it to your .env file."
    )