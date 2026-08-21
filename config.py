"""
Loads .env and exposes typed settings for the whole app.
"""
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Make sure you have a .env file with "
        "GROQ_API_KEY=your_key in the project root."
    )