
from pymongo import MongoClient

def get_mongo_client(uri="mongodb://localhost:27017/"):
    return MongoClient(uri)

def get_collection(db_name, collection_name, uri="mongodb://localhost:27017/"):
    client = get_mongo_client(uri)
    db = client[db_name]
    return db[collection_name]
