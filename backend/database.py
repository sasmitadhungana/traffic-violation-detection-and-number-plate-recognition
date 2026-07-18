import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient

# Load variables from .env file
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

class MockCollection:
    def __init__(self, filepath):
        self.filepath = filepath
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = []
        else:
            self.data = []

    def _save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        def default_serializer(o):
            if isinstance(o, datetime):
                return o.isoformat()
            return str(o)
        with open(self.filepath, "w") as f:
            json.dump(self.data, f, default=default_serializer, indent=2)

    def find_one(self, filter_dict):
        self._load()
        for doc in self.data:
            match = True
            for k, v in filter_dict.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                return dict(doc)
        return None

    def insert_one(self, doc):
        self._load()
        if "_id" not in doc:
            doc["_id"] = str(uuid.uuid4())
        self.data.append(doc)
        self._save()
        return doc

    def insert_many(self, docs):
        self._load()
        for doc in docs:
            if "_id" not in doc:
                doc["_id"] = str(uuid.uuid4())
            self.data.append(doc)
        self._save()
        return docs

    def update_one(self, filter_dict, update_dict):
        self._load()
        doc = self.find_one(filter_dict)
        if doc:
            for i, d in enumerate(self.data):
                if d.get("_id") == doc.get("_id"):
                    if "$set" in update_dict:
                        for k, v in update_dict["$set"].items():
                            self.data[i][k] = v
                    break
            self._save()

    def count_documents(self, filter_dict):
        self._load()
        if not filter_dict:
            return len(self.data)
        count = 0
        for doc in self.data:
            match = True
            for k, v in filter_dict.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                count += 1
        return count

    def find(self, filter_dict=None, projection=None):
        self._load()
        filter_dict = filter_dict or {}
        matched = []
        for doc in self.data:
            match = True
            for k, v in filter_dict.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                d = dict(doc)
                if projection and "_id" in projection and projection["_id"] == 0:
                    d.pop("_id", None)
                matched.append(d)
        
        class Cursor:
            def __init__(self, results):
                self.results = results
            def sort(self, field, direction=-1):
                reverse = True if direction == -1 else False
                self.results.sort(key=lambda x: x.get(field, ""), reverse=reverse)
                return self
            def limit(self, num):
                self.results = self.results[:num]
                return self
            def __iter__(self):
                return iter(self.results)
            def __len__(self):
                return len(self.results)
                
        return Cursor(matched)

    def aggregate(self, pipeline):
        self._load()
        from collections import defaultdict
        groups = defaultdict(int)
        for doc in self.data:
            val = doc.get("type", "Unknown")
            groups[val] += 1
        return [{"_id": k, "count": v} for k, v in groups.items()]

class MockDatabase:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.collections = {
            "users": MockCollection(os.path.join(data_dir, "users.json")),
            "video_analyses": MockCollection(os.path.join(data_dir, "video_analyses.json")),
            "violations": MockCollection(os.path.join(data_dir, "violations.json"))
        }

    def __getitem__(self, name):
        if name not in self.collections:
            data_dir = os.path.join(os.path.dirname(__file__), "data")
            self.collections[name] = MockCollection(os.path.join(data_dir, f"{name}.json"))
        return self.collections[name]

# Check connection
db = None
try:
    print("Attempting to connect to MongoDB...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    # Trigger selection to check if server is running
    client.admin.command('ping')
    db = client["atlvms_db"]
    print("Connected to MongoDB successfully!")
except Exception as e:
    print(f"MongoDB connection failed: {e}")
    print("=> Falling back to local JSON file-based database in F:\\TrafficViolationProject\\backend\\data\\")
    db = MockDatabase()

def get_db():
    return db
