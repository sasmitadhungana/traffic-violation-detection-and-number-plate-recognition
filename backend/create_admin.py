from passlib.context import CryptContext
from database import get_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

def create_admin():
    db = get_db()
    users_collection = db["users"]
    
    email = "admin@trafficguard.com"
    password = "password123"
    
    existing_user = users_collection.find_one({"email": email})
    if existing_user:
        print(f"User {email} already exists!")
        return
        
    hashed_password = get_password_hash(password)
    
    user_doc = {
        "email": email,
        "hashed_password": hashed_password,
        "role": "admin"
    }
    
    users_collection.insert_one(user_doc)
    print(f"Successfully created admin user: {email}")

if __name__ == "__main__":
    create_admin()
