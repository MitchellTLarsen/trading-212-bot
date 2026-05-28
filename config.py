import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("T212_API_KEY")
BASE_URL = "https://demo.trading212.com/api/v0"

HEADERS = {
    "Authorization": API_KEY,
    "Content-Type": "application/json",
}
