import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from db.models import Base
import sqlite_vec

@pytest.fixture(scope="session")
def db_engine():
    """Fixture for a database engine with an in-memory SQLite DB."""
    # Add check_same_thread=False to allow cross-thread access
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=None,  # Disable connection pooling for testing
        echo=False
    )
    
    @event.listens_for(engine, "connect")
    def load_vec_extension(dbapi_connection, connection_record):
        dbapi_connection.enable_load_extension(True)
        sqlite_vec.load(dbapi_connection)
        dbapi_connection.enable_load_extension(False)
    
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        conn.execute(
            text("CREATE VIRTUAL TABLE vss_embeddings USING vec0(vec FLOAT[768])")
        )
    
    yield engine
    engine.dispose()

@pytest.fixture(scope="function")
def db_session(db_engine):
    """Fixture for a database session, which gets rolled back after each test."""
    connection = db_engine.connect()
    transaction = connection.begin()

    Session = sessionmaker(bind=connection)
    session = Session()

    # Start a savepoint
    session.begin_nested()

    @event.listens_for(session, "after_transaction_end")
    def restart_savepoint(session, transaction):
        if transaction.nested and not transaction._parent.nested:
            session.expire_all()
            session.begin_nested()

    yield session

    # Clean up
    session.close()
    transaction.rollback()
    connection.close()