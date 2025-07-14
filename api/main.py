import sys
import os

# Set the correct DATABASE_URL for the api
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
database_path = os.path.join(project_root, "data", "rss.db")
os.environ["DATABASE_URL"] = f"sqlite:///{database_path}"

# Add parent directory to path to import db modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import exists, not_
from typing import List, Optional
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from db.database import SessionLocal, get_db
from db.models import Feed, Article, Interaction, ActionEnum
from services.feed_services import update_feeds
from services.article_services import fetch_articles, detect_topics_with_bertopic
from services.linucb_services import update_model, recalculate_scores
from api import schemas

app = FastAPI(
    title="HabitFeed",
    description="A local RSS reader with AI-powered recommendations driven by your personal habits",
    version="1.0.0"
)

# Define negative actions that should exclude articles from lists
NEGATIVE_ACTIONS = [ActionEnum.skip, ActionEnum.meh, ActionEnum.downvote]

@app.get("/feeds", response_model=List[schemas.FeedInfoSchema])
def get_all_feeds(db: Session = Depends(get_db)):
    """Get all feeds from the database"""
    feeds = db.query(Feed).order_by(Feed.title).all()
    return feeds

@app.get("/feeds/{feed_id}", response_model=schemas.FeedSchema)
def get_feed_by_id(feed_id: int, db: Session = Depends(get_db)):
    """Get a specific feed by ID"""
    feed = db.query(Feed).filter(Feed.feed_id == feed_id).first()
    if feed is None:
        raise HTTPException(status_code=404, detail="Feed not found")
    return feed

@app.get("/feeds/{feed_id}/articles", response_model=List[schemas.ArticleSchema])
def get_articles_by_feed(feed_id: int, limit: Optional[int] = None, db: Session = Depends(get_db)):
    """Get all articles for a specific feed, excluding those with negative actions"""
    # Query articles that don't have any negative interactions
    query = db.query(Article).filter(
        Article.feed_id == feed_id,
        not_(exists().where(
            Interaction.article_id == Article.article_id,
            Interaction.action.in_(NEGATIVE_ACTIONS)
        ))
    ).order_by(Article.vector_similarity_score.desc().nullslast(), Article.pub_date.desc())
    
    if limit:
        query = query.limit(limit)
    articles = query.all()
    return articles

@app.get("/articles/recent", response_model=List[schemas.ArticleSchema])
def get_recent_articles(limit: int = 10, db: Session = Depends(get_db)):
    """Get recent articles across all feeds, excluding those with negative actions"""
    # Query articles that don't have any negative interactions
    articles = db.query(Article).filter(
        Article.title.isnot(None),
        not_(exists().where(
            Interaction.article_id == Article.article_id,
            Interaction.action.in_(NEGATIVE_ACTIONS)
        ))
    ).order_by(Article.pub_date.desc()).limit(limit).all()
    return articles

@app.get("/articles/recommended", response_model=List[schemas.ArticleSchema])
def get_recommended_articles(limit: int = 10, db: Session = Depends(get_db)):
    """Get recommended articles across all feeds, ordered by vector similarity score then date, excluding those with negative actions"""
    # Query articles that don't have any negative interactions
    articles = db.query(Article).filter(
        Article.title.isnot(None),
        Article.vector_similarity_score.isnot(None),
        not_(exists().where(
            Interaction.article_id == Article.article_id,
            Interaction.action.in_(NEGATIVE_ACTIONS)
        ))
    ).order_by(Article.vector_similarity_score.desc(), Article.pub_date.desc()).limit(limit).all()
    return articles

@app.get("/articles/recommended-linucb", response_model=List[schemas.ArticleSchema])
def get_recommended_articles_linucb(limit: int = 20, db: Session = Depends(get_db)):
    """Get recommended articles ranked by LinUCB score, excluding those with negative actions"""
    # Query articles that don't have any negative interactions and have LinUCB scores
    articles = db.query(Article).filter(
        Article.title.isnot(None),
        Article.linucb_score.isnot(None),
        not_(exists().where(
            Interaction.article_id == Article.article_id,
            Interaction.action.in_(NEGATIVE_ACTIONS)
        ))
    ).order_by(Article.linucb_score.desc(), Article.pub_date.desc()).limit(limit).all()
    
    return articles

@app.get("/articles/{article_id}", response_model=schemas.ArticleSchema)
def get_article_by_id(article_id: int, db: Session = Depends(get_db)):
    """Get a specific article by ID"""
    article = db.query(Article).filter(Article.article_id == article_id).first()
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found")
    return article

@app.get("/articles/{article_id}/actions", response_model=List[schemas.InteractionSchema])
def get_actions_for_article_by_id(article_id: int, db: Session = Depends(get_db)):
    """Get all actions/interactions for a specific article by ID"""
    # Check if article exists
    article = db.query(Article).filter(Article.article_id == article_id).first()
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found")
    
    # Get all interactions for this article, ordered by timestamp (most recent first)
    interactions = db.query(Interaction).filter(
        Interaction.article_id == article_id
    ).order_by(Interaction.event_ts.desc()).all()
    
    return interactions

@app.get("/articles/action_feed/{action}", response_model=List[schemas.ArticleSchema])
def get_articles_by_action(action: str, db: Session = Depends(get_db)):
    """Get all articles that have a specific action, ordered by action timestamp (most recent first)"""
    # Validate action is in ActionEnum
    try:
        action_enum = ActionEnum(action)
    except ValueError:
        raise HTTPException(status_code=406, detail=f"Action '{action}' is not supported")
    
    # Query articles that have the specified action
    # Join with Interaction table and order by interaction timestamp
    articles = db.query(Article).join(Interaction).filter(
        Interaction.action == action_enum
    ).order_by(Interaction.event_ts.desc()).all()
    
    return articles

@app.post("/feeds", response_model=schemas.FeedInfoSchema, status_code=201)
def add_feed(feed: schemas.FeedCreate, db: Session = Depends(get_db)):
    """Add a new feed to the database"""
    existing_feed = db.query(Feed).filter(Feed.feed_url == feed.feed_url).first()
    if existing_feed:
        raise HTTPException(status_code=400, detail="Feed already exists")
    
    new_feed = Feed(feed_url=feed.feed_url)
    db.add(new_feed)
    db.commit()
    db.refresh(new_feed)
    return new_feed

@app.post("/articles/{article_id}/actions", response_model=schemas.ActionResponse, status_code=201)
def record_article_action(
    article_id: int, 
    action_data: schemas.ActionRecord, 
    db: Session = Depends(get_db)
):
    """Record an action on an article"""
    # Check if article exists
    article = db.query(Article).filter(Article.article_id == article_id).first()
    if article is None:
        raise HTTPException(status_code=404, detail="Article not found")
    
    # Validate action is in ActionEnum
    try:
        action_enum = ActionEnum(action_data.action)
    except ValueError:
        raise HTTPException(status_code=406, detail=f"Action '{action_data.action}' is not supported")
    
    # Get existing interactions for this article
    existing_interactions = db.query(Interaction).filter(Interaction.article_id == article_id).all()
    
    has_negative_interactions = any(interaction.action in NEGATIVE_ACTIONS for interaction in existing_interactions)
    has_click_interactions = any(interaction.action == ActionEnum.click for interaction in existing_interactions)
    
    # Determine the actual action to record
    actual_action = action_enum
    
    # If incoming action is 'click' and there are existing negative interactions,
    # convert to 'click_archived'
    if action_enum == ActionEnum.click and has_negative_interactions:
        actual_action = ActionEnum.click_archived
    
    # If incoming action is negative, convert existing 'click' events to 'click_archived'
    if action_enum in NEGATIVE_ACTIONS and has_click_interactions:
        click_interactions = db.query(Interaction).filter(
            Interaction.article_id == article_id,
            Interaction.action == ActionEnum.click
        ).all()
        
        for interaction in click_interactions:
            interaction.action = ActionEnum.click_archived
        
    # Set vector_similarity_score to -2 if action is not 'dwell'
    if actual_action != ActionEnum.dwell:
        article.vector_similarity_score = -2.0
    
    # Create the new interaction
    new_interaction = Interaction(
        article_id=article_id,
        action=actual_action,
        action_duration_ms=action_data.action_duration_ms
    )
    
    db.add(new_interaction)
    db.commit()
    
    # Update the LinUCB model after recording the action
    try:
        update_model(db, article_id, actual_action, action_data.action_duration_ms)
    except Exception as e:
        # Log error but don't fail the request
        print(f"Error updating LinUCB model: {e}")
    
    return schemas.ActionResponse(
        message="Action recorded successfully",
        recorded_action=actual_action.value,
        article_id=article_id
    )

#########################
# SERVICE ENDPOINTS
#########################

@app.post("/feeds/update")
def update_all_feeds(db: Session = Depends(get_db)):
    """Update all feeds by calling the feed service"""
    try:
        update_feeds(db)
        return {"message": "Feeds update started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating feeds: {e}")

@app.post("/articles/fetch")
def fetch_all_articles(db: Session = Depends(get_db)):
    """Fetch article content and create embeddings by calling the article service"""
    try:
        fetch_articles(db)
        return {"message": "Article fetching started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching articles: {e}")

@app.post("/articles/detect-topics")
def detect_topics(db: Session = Depends(get_db)):
    """Detect topics for articles using BERTopic"""
    try:
        results = detect_topics_with_bertopic(db)
        return {
            "message": "Topic detection completed",
            "articles_processed": results["articles_processed"],
            "topics_created": results["topics_created"],
            "total_topics_found": results["total_topics_found"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting topics: {e}")

@app.post("/articles/recalculate-linucb-scores")
def recalculate_linucb_scores(top_n: int = 100, db: Session = Depends(get_db)):
    """Recalculate LinUCB scores for articles"""
    try:
        updated_count = recalculate_scores(db, top_n)
        return {
            "message": "LinUCB scores recalculated",
            "articles_updated": updated_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recalculating LinUCB scores: {e}")

#########################
# SSE SERVICE ENDPOINTS
#########################

async def generate_feed_update_events(db: Session):
    """Generator for SSE events during feed updates"""
    progress_queue = asyncio.Queue()
    
    def progress_callback(current: int, total: int, message: str):
        try:
            asyncio.get_event_loop().call_soon_threadsafe(
                progress_queue.put_nowait, (current, total, message)
            )
        except RuntimeError:
            # Handle case where event loop is not running
            pass
    
    # Run the update_feeds in a thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    
    # Start the feed update task
    future = loop.run_in_executor(executor, update_feeds, db, progress_callback)
    
    try:
        while True:
            try:
                # Check if the task is done
                if future.done():
                    # Get any remaining items from the queue
                    while not progress_queue.empty():
                        current, total, message = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                        event_data = {
                            "current": current,
                            "total": total,
                            "message": message,
                            "completed": current >= total
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"
                    break
                else:
                    # Get progress updates
                    current, total, message = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                    event_data = {
                        "current": current,
                        "total": total,
                        "message": message,
                        "completed": current >= total
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
            except asyncio.TimeoutError:
                # Send keep-alive
                yield f"data: {json.dumps({'type': 'keep-alive'})}\n\n"
                continue
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        executor.shutdown(wait=False)

async def generate_article_fetch_events(db: Session):
    """Generator for SSE events during article fetching"""
    progress_queue = asyncio.Queue()
    
    def progress_callback(current: int, total: int, message: str):
        try:
            asyncio.get_event_loop().call_soon_threadsafe(
                progress_queue.put_nowait, (current, total, message)
            )
        except RuntimeError:
            # Handle case where event loop is not running
            pass
    
    # Run the fetch_articles in a thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    
    # Start the article fetch task
    future = loop.run_in_executor(executor, fetch_articles, db, progress_callback)
    
    try:
        while True:
            try:
                # Check if the task is done
                if future.done():
                    # Get any remaining items from the queue
                    while not progress_queue.empty():
                        current, total, message = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                        event_data = {
                            "current": current,
                            "total": total,
                            "message": message,
                            "completed": current >= total
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"
                    break
                else:
                    # Get progress updates
                    current, total, message = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                    event_data = {
                        "current": current,
                        "total": total,
                        "message": message,
                        "completed": current >= total
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
            except asyncio.TimeoutError:
                # Send keep-alive
                yield f"data: {json.dumps({'type': 'keep-alive'})}\n\n"
                continue
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        executor.shutdown(wait=False)

async def generate_topic_detection_events(db: Session):
    """Generator for SSE events during topic detection"""
    progress_queue = asyncio.Queue()
    
    def progress_callback(current: int, total: int, message: str):
        try:
            asyncio.get_event_loop().call_soon_threadsafe(
                progress_queue.put_nowait, (current, total, message)
            )
        except RuntimeError:
            # Handle case where event loop is not running
            pass
    
    # Run the topic detection in a thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    
    # Start the topic detection task
    future = loop.run_in_executor(executor, detect_topics_with_bertopic, db, 10, progress_callback)

async def generate_linucb_recalculation_events(db: Session, top_n: int = 100):
    """Generator for SSE events during LinUCB score recalculation"""
    progress_queue = asyncio.Queue()
    
    def progress_callback(current: int, total: int, message: str):
        try:
            asyncio.get_event_loop().call_soon_threadsafe(
                progress_queue.put_nowait, (current, total, message)
            )
        except RuntimeError:
            # Handle case where event loop is not running
            pass
    
    # Run the recalculate_scores in a thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    
    # Start the recalculation task
    future = loop.run_in_executor(executor, recalculate_scores, db, top_n, progress_callback)
    
    try:
        while True:
            try:
                # Check if the task is done
                if future.done():
                    # Get any remaining items from the queue
                    while not progress_queue.empty():
                        current, total, message = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                        event_data = {
                            "current": current,
                            "total": total,
                            "message": message,
                            "completed": current >= total
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"
                    
                    # Send final results
                    try:
                        updated_count = future.result()
                        final_event = {
                            "type": "completion",
                            "articles_updated": updated_count,
                            "completed": True
                        }
                        yield f"data: {json.dumps(final_event)}\n\n"
                    except Exception as e:
                        error_event = {"type": "error", "error": str(e)}
                        yield f"data: {json.dumps(error_event)}\n\n"
                    break
                else:
                    # Get progress updates
                    current, total, message = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                    event_data = {
                        "current": current,
                        "total": total,
                        "message": message,
                        "completed": current >= total
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
            except asyncio.TimeoutError:
                # Send keep-alive
                yield f"data: {json.dumps({'type': 'keep-alive'})}\n\n"
                continue
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        executor.shutdown(wait=False)

@app.get("/feeds/update/stream")
async def update_all_feeds_stream(db: Session = Depends(get_db)):
    """Update all feeds with real-time progress via SSE"""
    return StreamingResponse(
        generate_feed_update_events(db),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

@app.get("/articles/fetch/stream")
async def fetch_all_articles_stream(db: Session = Depends(get_db)):
    """Fetch article content and create embeddings with real-time progress via SSE"""
    return StreamingResponse(
        generate_article_fetch_events(db),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

@app.get("/articles/detect-topics/stream")
async def detect_topics_stream(db: Session = Depends(get_db)):
    """Detect topics for articles using BERTopic with real-time progress via SSE"""
    return StreamingResponse(
        generate_topic_detection_events(db),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

@app.get("/articles/recalculate-linucb-scores/stream")
async def recalculate_linucb_scores_stream(top_n: int = 100, db: Session = Depends(get_db)):
    """Recalculate LinUCB scores with real-time progress via SSE"""
    return StreamingResponse(
        generate_linucb_recalculation_events(db, top_n),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    ) 