import enum

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    TIMESTAMP,
    Text,
    ForeignKey,
    Table,
    Float,
    Enum,
    LargeBinary,
    Index,
    text,
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

from .database import engine

Base = declarative_base()

class ActionEnum(str, enum.Enum):
    click_archived = "click_archived"
    click = "click"
    dwell = "dwell"
    upvote = "upvote"
    downvote = "downvote"
    save = "save"
    meh = "meh"
    skip = "skip"

article_topic_association = Table(
    'article_topic', Base.metadata,
    Column('article_id', Integer, ForeignKey('article.article_id'), primary_key=True),
    Column('topic_id', Integer, ForeignKey('topic.topic_id'), primary_key=True),
    Column('confidence', Float)
)

class Feed(Base):
    __tablename__ = 'feed'
    feed_id = Column(Integer, primary_key=True, autoincrement=True)
    feed_url = Column(String, unique=True, nullable=False)
    title = Column(String)
    last_checked = Column(TIMESTAMP)
    articles = relationship("Article", back_populates="feed")

class Article(Base):
    __tablename__ = 'article'
    article_id = Column(Integer, primary_key=True, autoincrement=True)
    feed_id = Column(Integer, ForeignKey('feed.feed_id'))
    guid = Column(String, unique=True)
    url = Column(String)
    # NOTE: if title not set, we assume it is "empty" and needs to be fetched
    title = Column(String)
    rss_description = Column(String)
    pub_date = Column(TIMESTAMP)
    # NOTE: if language is "SKIP", we don't try to re-fetch ever
    language = Column(String(10))
    content_hash = Column(String(40)) # SHA-1 is 40 hex chars
    full_text = Column(Text)
    vector_similarity_score = Column(Float)
    linucb_score = Column(Float)
    feed = relationship("Feed", back_populates="articles")
    embedding = relationship("Embedding", uselist=False, back_populates="article")
    interactions = relationship("Interaction", back_populates="article")
    topics = relationship("Topic", secondary=article_topic_association, back_populates="articles")


class Embedding(Base):
    __tablename__ = 'embedding'
    article_id = Column(Integer, ForeignKey('article.article_id'), primary_key=True)
    vec = Column(LargeBinary) # Storing as a blob of FLOAT32s

    article = relationship("Article", back_populates="embedding")

class Interaction(Base):
    __tablename__ = 'interaction'
    int_id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(Integer, ForeignKey('article.article_id'))
    event_ts = Column(TIMESTAMP, server_default=func.now())
    action_duration_ms = Column(Integer)
    action = Column(Enum(ActionEnum))

    article = relationship("Article", back_populates="interactions")

    __table_args__ = (
        Index('ix_interaction_article_ts', "article_id", "event_ts"),
    )

class Topic(Base):
    __tablename__ = 'topic'
    topic_id = Column(Integer, primary_key=True, autoincrement=True)
    topic_data = Column(Text)
    topic_type = Column(Integer)
    label = Column(String, unique=True)

    articles = relationship("Article", secondary=article_topic_association, back_populates="topics")


class UserProfile(Base):
    __tablename__ = 'user_profile'
    profile_id = Column(Integer, primary_key=True, autoincrement=True)
    interest_vector_numerator = Column(LargeBinary)
    interest_vector_denominator = Column(LargeBinary)
    negative_interest_vector_numerator = Column(LargeBinary)
    negative_interest_vector_denominator = Column(LargeBinary)
    interest_vectors_updated_at = Column(TIMESTAMP)


class LinucbState(Base):
    __tablename__ = 'linucb_state'
    user_id = Column(Integer, primary_key=True)
    A = Column(LargeBinary)  # 16×16 matrix, row-major Float32 (feature-based)
    b = Column(LargeBinary)  # 16×1 vector (feature-based)
    last_update = Column(TIMESTAMP, server_default=func.now())


def init_db():
    """Initializes the database, creating tables and the vec index."""
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        try:
            conn.execute(
                text("CREATE VIRTUAL TABLE IF NOT EXISTS vss_embeddings USING vec0(vec FLOAT[768])")
            )
            print("VSS virtual table created (or already exists).")
        except Exception as e:
            print(f"An error occurred while creating VSS virtual table: {e}")
        conn.commit()