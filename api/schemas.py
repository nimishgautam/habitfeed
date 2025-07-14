from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ArticleSchema(BaseModel):
    article_id: int
    feed_id: int
    guid: str
    url: str
    title: Optional[str] = None
    rss_description: Optional[str] = None
    pub_date: Optional[datetime] = None
    language: Optional[str] = None
    full_text: Optional[str] = None
    vector_similarity_score: Optional[float] = None

    class Config:
        from_attributes = True

class FeedSchema(BaseModel):
    feed_id: int
    feed_url: str
    title: Optional[str] = None
    last_checked: Optional[datetime] = None
    articles: List[ArticleSchema] = []

    class Config:
        from_attributes = True

class FeedInfoSchema(BaseModel):
    feed_id: int
    feed_url: str
    title: Optional[str] = None
    last_checked: Optional[datetime] = None

    class Config:
        from_attributes = True

class FeedCreate(BaseModel):
    feed_url: str

class ActionRecord(BaseModel):
    action: str
    action_duration_ms: Optional[int] = None

class ActionResponse(BaseModel):
    message: str
    recorded_action: str
    article_id: int

class InteractionSchema(BaseModel):
    int_id: int
    article_id: int
    event_ts: datetime
    action_duration_ms: Optional[int] = None
    action: str

    class Config:
        from_attributes = True 