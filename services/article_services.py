import hashlib
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse
from loguru import logger
from typing import Optional, Callable

from goose3 import Goose
from sqlalchemy import and_, text, exists
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import numpy as np
from bertopic import BERTopic

from db.models import Article, Embedding, Interaction, ActionEnum, Topic, article_topic_association

# Using all-mpnet-base-v2 for 768-dimensional embeddings to match schema
model = SentenceTransformer('all-mpnet-base-v2')


class DomainRateLimiter:
    """Rate limiter that enforces delays per domain to avoid hitting rate limits."""
    
    def __init__(self, delay_per_domain=2.0):
        self.last_request_time = defaultdict(float)
        self.delay = delay_per_domain
    
    def wait_if_needed(self, url):
        """Wait if necessary to respect rate limits for the domain of the given URL."""
        domain = urlparse(url).netloc
        now = time.time()
        time_since_last = now - self.last_request_time[domain]
        
        if time_since_last < self.delay:
            sleep_time = self.delay - time_since_last
            logger.info(f"Rate limiting {domain}: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        self.last_request_time[domain] = time.time()


# Global rate limiter instance
rate_limiter = DomainRateLimiter(delay_per_domain=2.0)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(Exception)
)
def _fetch_article_with_retry(article: Article, db: Session):
    """
    Internal function that performs the actual article fetching with retry logic.
    """
    # Apply domain-based rate limiting
    rate_limiter.wait_if_needed(article.url)
    
    g = Goose()
    g_article = g.extract(url=article.url)
    
    # Check if we got meaningful content
    if not g_article.cleaned_text and not g_article.title:
        raise Exception("Empty response - possible rate limit or blocked request")
    
    article.title = g_article.title
    article.full_text = g_article.cleaned_text

    if g_article.publish_datetime_utc:
        article.pub_date = g_article.publish_datetime_utc
    elif not article.pub_date:
        article.pub_date = datetime.now(timezone.utc)
    
    if article.full_text:
        article.content_hash = hashlib.sha1(article.full_text.encode('utf-8')).hexdigest()

    db.add(article)
    return True


def fetch_article(article: Article, db: Session):
    """
    Fetches a single article, populating title, pub_date, content_hash and full_text.
    Uses retry logic and marks articles as 'SKIP' if all attempts fail.
    """
    try:
        return _fetch_article_with_retry(article, db)
    except Exception as e:
        logger.error(f"Failed to fetch article {article.url} after all retry attempts: {e}")
        # Mark article as 'SKIP' so we don't try again
        article.language = 'SKIP'
        db.add(article)
        db.commit()
        return False

def create_embedding(article: Article, db: Session):
    """
    Creates an embedding for the article and stores it in the 'embedding' and 'vss_embeddings' tables.
    """
    if not article.full_text or not article.article_id:
        logger.warning(f"Article {article.article_id or article.url} has no text or ID, skipping embedding.")
        return False
    
    # Check if embedding already exists
    existing_embedding = db.query(Embedding).filter(Embedding.article_id == article.article_id).first()
    if existing_embedding:
        logger.info(f"Embedding for article {article.article_id} already exists, skipping.")
        return True

    try:
        embedding_vector = model.encode(article.full_text, convert_to_tensor=False)
        embedding_blob = embedding_vector.astype(np.float32).tobytes()

        # Insert into 'embedding' table
        new_embedding = Embedding(article_id=article.article_id, vec=embedding_blob)
        db.add(new_embedding)

        # Insert into 'vss_embeddings' FTS table for searching
        # The rowid in vss_embeddings must correspond to article_id
        stmt = text("INSERT INTO vss_embeddings (rowid, vec) VALUES (:article_id, :vec)")
        db.execute(stmt, {'article_id': article.article_id, 'vec': embedding_blob})
        
        return True
    except Exception as e:
        logger.error(f"Error creating embedding for article {article.article_id}: {e}")
        db.rollback()
        return False


def fetch_articles(db: Session, progress_callback: Optional[Callable[[int, int, str], None]] = None):
    """
    Fetches all articles where title is not set and language is not explicitly "SKIP".
    
    Args:
        db: Database session
        progress_callback: Optional callback function that takes (current, total, message) parameters
    """
    articles_to_fetch = db.query(Article).filter(
        and_(
            Article.title.is_(None),
            (Article.language.is_(None) | (Article.language != 'SKIP'))
        )
    ).all()

    total_articles = len(articles_to_fetch)
    logger.info(f"Fetching {total_articles} articles")

    for i, article in enumerate(articles_to_fetch, 1):
        if progress_callback:
            progress_callback(i, total_articles, f"Processing article: {article.url}")
        
        if fetch_article(article, db):
            db.commit()
            if create_embedding(article, db):
                db.commit()
    
    if progress_callback:
        progress_callback(total_articles, total_articles, "All articles processed successfully")
    
    db.commit()


def detect_topics_with_bertopic(db: Session, min_articles: int = 10, progress_callback: Optional[Callable[[int, int, str], None]] = None):
    """
    Uses BERTopic to detect topics for articles that don't have any topics assigned yet.
    Stores the top 5 topics and their probabilities for each article.
    
    Args:
        db: Database session
        min_articles: Minimum number of articles needed to run topic detection (default: 10)
        progress_callback: Optional callback function that takes (current, total, message) parameters
    
    Returns:
        dict: Results containing number of articles processed and topics created
    """
    # Find articles that don't have any topics assigned yet
    articles_without_topics = db.query(Article).filter(
        and_(
            Article.full_text.isnot(None),
            ~exists().where(article_topic_association.c.article_id == Article.article_id)
        )
    ).all()
    
    if len(articles_without_topics) < min_articles:
        logger.info(f"Only {len(articles_without_topics)} articles without topics found. Need at least {min_articles} to run topic detection.")
        return {"articles_processed": 0, "topics_created": 0}
    
    logger.info(f"Found {len(articles_without_topics)} articles without topics. Starting BERTopic analysis...")
    
    if progress_callback:
        progress_callback(0, len(articles_without_topics), "Loading embeddings...")
    
    # Load embeddings for these articles
    article_embeddings = []
    article_ids = []
    article_texts = []
    
    for i, article in enumerate(articles_without_topics):
        embedding_record = db.query(Embedding).filter(Embedding.article_id == article.article_id).first()
        if embedding_record and embedding_record.vec:
            # Convert blob back to numpy array
            embedding_vector = np.frombuffer(embedding_record.vec, dtype=np.float32)
            article_embeddings.append(embedding_vector)
            article_ids.append(article.article_id)
            article_texts.append(article.full_text)
        
        if progress_callback and i % 50 == 0:
            progress_callback(i, len(articles_without_topics), f"Loading embeddings... ({i}/{len(articles_without_topics)})")
    
    if not article_embeddings:
        logger.warning("No embeddings found for articles without topics.")
        return {"articles_processed": 0, "topics_created": 0}
    
    logger.info(f"Loaded {len(article_embeddings)} embeddings. Running BERTopic...")
    
    if progress_callback:
        progress_callback(len(articles_without_topics), len(articles_without_topics), "Running BERTopic analysis...")
    
    # Initialize BERTopic with pre-computed embeddings
    # We'll use the embeddings we already have instead of re-computing them
    topic_model = BERTopic(
        verbose=True,
        calculate_probabilities=True,
        nr_topics="auto"  # Let BERTopic determine optimal number of topics
    )
    
    # Fit the model and get topic assignments and probabilities
    topics, probabilities = topic_model.fit_transform(article_texts, embeddings=np.array(article_embeddings))
    
    logger.info(f"BERTopic analysis complete. Found {len(set(topics))} topics.")
    
    if progress_callback:
        progress_callback(len(articles_without_topics), len(articles_without_topics), "Storing topic assignments...")
    
    # Get topic info from BERTopic
    topic_info = topic_model.get_topic_info()
    
    # Create Topic records for new topics and build a mapping
    existing_topics = {}
    topics_created = 0
    
    for topic_id in set(topics):
        if topic_id == -1:  # BERTopic's "no topic" label
            topic_label = "No Topic"
            topic_data = "Outlier topic - no clear theme"
        else:
            # Get the topic words/representation
            topic_words = topic_model.get_topic(topic_id)
            topic_label = f"Topic_{topic_id}"
            topic_data = ", ".join([word for word, score in topic_words[:10]])  # Top 10 words
        
        # Check if this topic already exists in database
        existing_topic = db.query(Topic).filter(Topic.label == topic_label).first()
        if existing_topic:
            existing_topics[topic_id] = existing_topic.topic_id
        else:
            # Create new topic
            new_topic = Topic(
                label=topic_label,
                topic_data=topic_data,
                topic_type=1  # BERTopic type
            )
            db.add(new_topic)
            db.flush()  # Get the ID without committing
            existing_topics[topic_id] = new_topic.topic_id
            topics_created += 1
    
    # Store topic assignments with probabilities
    articles_processed = 0
    
    for i, (article_id, article_topic, article_probs) in enumerate(zip(article_ids, topics, probabilities)):
        # Get top 5 topics with highest probabilities for this article
        if article_probs is not None:
            # Get indices of top 5 probabilities
            top_indices = np.argsort(article_probs)[-5:][::-1]  # Top 5 in descending order
            top_topics = [(idx, article_probs[idx]) for idx in top_indices if article_probs[idx] > 0.01]  # Only store if prob > 1%
        else:
            # Fallback if probabilities aren't available - just use the assigned topic
            top_topics = [(article_topic, 1.0)]
        
        # Store each topic assignment
        for topic_idx, confidence in top_topics:
            if topic_idx in existing_topics:
                # Insert into article_topic association table
                stmt = text("""
                    INSERT INTO article_topic (article_id, topic_id, confidence) 
                    VALUES (:article_id, :topic_id, :confidence)
                """)
                db.execute(stmt, {
                    'article_id': article_id,
                    'topic_id': existing_topics[topic_idx],
                    'confidence': float(confidence)
                })
        
        articles_processed += 1
        
        if progress_callback and i % 50 == 0:
            progress_callback(i, len(article_ids), f"Storing topic assignments... ({i}/{len(article_ids)})")
    
    db.commit()
    
    logger.info(f"Topic detection complete. Processed {articles_processed} articles, created {topics_created} new topics.")
    
    if progress_callback:
        progress_callback(len(article_ids), len(article_ids), f"Complete! Processed {articles_processed} articles, created {topics_created} topics.")
    
    return {
        "articles_processed": articles_processed,
        "topics_created": topics_created,
        "total_topics_found": len(set(topics))
    }


def delete_old_articles(db: Session, days_old: int = 90):
    """
    Deletes articles whose publication date is older than the specified number of days,
    unless they have 'upvote', 'save', 'downvote', or 'skip' actions associated with them.
    
    Args:
        db: Database session
        days_old: Number of days after which articles should be considered old (default: 90)
    
    Returns:
        int: Number of articles deleted
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
    
    # Find articles that are older than the cutoff date
    # but exclude those that have the specified actions
    protected_actions = [ActionEnum.upvote, ActionEnum.save, ActionEnum.downvote, ActionEnum.skip]
    
    # Subquery to find articles with protected actions
    protected_articles_subquery = db.query(Interaction.article_id).filter(
        Interaction.action.in_(protected_actions)
    ).distinct()
    
    # Query for articles to delete: old articles without protected actions
    articles_to_delete = db.query(Article).filter(
        and_(
            Article.pub_date < cutoff_date,
            ~Article.article_id.in_(protected_articles_subquery)
        )
    ).all()
    
    deleted_count = len(articles_to_delete)
    
    if deleted_count > 0:
        logger.info(f"Deleting {deleted_count} articles older than {days_old} days")
        
        # Delete associated embeddings first (due to foreign key constraints)
        for article in articles_to_delete:
            # Delete from vss_embeddings table
            db.execute(
                text("DELETE FROM vss_embeddings WHERE rowid = :article_id"),
                {'article_id': article.article_id}
            )
            
            # Delete from embedding table
            db.query(Embedding).filter(Embedding.article_id == article.article_id).delete()
            
            # Delete from interaction table
            db.query(Interaction).filter(Interaction.article_id == article.article_id).delete()
            
            # Delete the article itself
            db.delete(article)
        
        db.commit()
        logger.info(f"Successfully deleted {deleted_count} old articles and their associated data")
    else:
        logger.info(f"No articles older than {days_old} days found for deletion")
    
    return deleted_count 