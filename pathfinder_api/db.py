from pymongo import MongoClient
from dotenv import load_dotenv
import os
load_dotenv()
db_uri = os.getenv("MONGODB_URI")
client = MongoClient(db_uri)
db = client["graph"]
collection = db["graph"]
