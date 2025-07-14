import os
import sqlite3
import sqlite_vec

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
database_path = os.path.join(project_root, "data", "rss.db")
DEFAULT_DB_PATH = f"sqlite:///{database_path}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_PATH)

engine_args = {}
if DATABASE_URL.startswith("sqlite"):
    engine_args["connect_args"] = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL, **engine_args
)

# Load vec extension
@event.listens_for(engine, "connect")
def load_vec_extension(dbapi_connection, connection_record):
    """
    Enable the vec extension on a new connection.
    See: https://github.com/asg017/sqlite-vec
    """
    dbapi_connection.enable_load_extension(True)
    sqlite_vec.load(dbapi_connection)
    dbapi_connection.enable_load_extension(False)
    print("sqlite-vec extension loaded.")


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 