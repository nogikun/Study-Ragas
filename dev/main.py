import os
from dotenv import load_dotenv

load_dotenv()
endpoint = os.environ.get("AZURE_ENDPOINT")
key = os.environ.get("AZURE_KEY")

print(endpoint)
print(key)