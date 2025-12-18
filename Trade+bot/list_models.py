import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

try:
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    pager = client.models.list()
    print("Available models:")
    for model in pager:
        print(f"- {model.name}")
except Exception as e:
    print(f"Error listing models: {e}")
