import feedparser
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from loguru import logger
from typing import Optional, Callable

from db.models import Feed, Article

def update_feed(feed: Feed, db: Session):
    """
    Updates a single feed, adding new articles to the database.
    """
    try:
        parsed_feed = feedparser.parse(feed.feed_url)

        # Update feed title if it's not set
        if not feed.title and parsed_feed.feed.title:
            feed.title = parsed_feed.feed.title

        for entry in parsed_feed.entries:
            guid = entry.get("guid", entry.link)
            if not guid:
                # Can't process articles without a unique identifier
                continue
            
            # Check if article already exists
            exists = db.query(Article).filter(Article.guid == guid).first()
            if not exists:
                new_article = Article(
                    feed_id=feed.feed_id,
                    guid=guid,
                    url=entry.link,
                    rss_description=entry.get("description"),
                    # NOTE: we intentionally don't get title here, we save it for the Article processing
                    # We actually use no title as a check for the article needing processing!!
                    #title=entry.get("title"),
                    pub_date=datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else None,
                )
                db.add(new_article)
        
        feed.last_checked = datetime.now(timezone.utc)
        db.add(feed)
        db.commit()
    except Exception as e:
        logger.error(f"Error updating feed {feed.feed_url}: {e}")
        db.rollback()


def update_feeds(db: Session, progress_callback: Optional[Callable[[int, int, str], None]] = None):
    """
    Updates all feeds in the database.
    
    Args:
        db: Database session
        progress_callback: Optional callback function that takes (current, total, message) parameters
    """
    feeds = db.query(Feed).all()
    total_feeds = len(feeds)
    
    for i, feed in enumerate(feeds, 1):
        if progress_callback:
            progress_callback(i, total_feeds, f"Updating feed: {feed.title or feed.feed_url}")
        update_feed(feed, db)
    
    if progress_callback:
        progress_callback(total_feeds, total_feeds, "All feeds updated successfully") 