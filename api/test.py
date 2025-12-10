import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path='.env')

# Retrieve the NEUPHONIC_API_KEY from environment variables
api_key = os.getenv('NEUPHONIC_API_KEY')

if api_key is None:
    raise ValueError("NEUPHONIC_API_KEY not found in environment variables")

print(f"Loaded NEUPHONIC_API_KEY: {api_key}")
print(f"Loaded API Key: {api_key}")
