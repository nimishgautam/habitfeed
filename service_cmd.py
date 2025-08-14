#!/usr/bin/env python3
"""
Service command utility for HabitFeed application.
Provides convenient CLI access to various service operations.
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy.orm import sessionmaker
from db.database import engine
from db.models import init_db
from services.article_services import detect_topics_with_bertopic, fetch_articles, delete_old_articles
from services.feed_services import update_feeds
from services.embedding_services import calculate_profile_embeddings, calculate_scores_for_recent_uninteracted_articles
from services.linucb_services import recalculate_scores
from loguru import logger

def progress_callback(current: int, total: int, message: str):
    """Simple progress callback that prints updates."""
    percentage = (current / total) * 100 if total > 0 else 0
    print(f"[{percentage:.1f}%] {message}")

def cmd_create_db():
    """Create and initialize the database."""
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Created data directory.")

    print("Initializing database...")
    init_db()
    print("Database initialized successfully.")

def cmd_detect_topics(min_articles: int = 5):
    """Run BERTopic topic detection on articles without topics."""
    logger.info("Starting BERTopic topic detection...")

    # Create database session
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        # Run topic detection
        results = detect_topics_with_bertopic(
            db=db,
            min_articles=min_articles,
            progress_callback=progress_callback
        )

        print("\n" + "="*50)
        print("TOPIC DETECTION RESULTS")
        print("="*50)
        print(f"Articles processed: {results['articles_processed']}")
        print(f"New topics created: {results['topics_created']}")
        print(f"Total topics found: {results['total_topics_found']}")
        print("="*50)

        if results['articles_processed'] > 0:
            logger.info("Topic detection completed successfully!")
        else:
            logger.info("No articles were processed. Make sure you have articles with embeddings but no topics.")

    except Exception as e:
        logger.error(f"Error during topic detection: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def cmd_update_feeds():
    """Update all RSS feeds."""
    logger.info("Starting feed updates...")

    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        update_feeds(db=db, progress_callback=progress_callback)
        logger.info("Feed updates completed successfully!")
    except Exception as e:
        logger.error(f"Error during feed updates: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def cmd_fetch_articles():
    """Fetch and process articles from feeds."""
    logger.info("Starting article fetching...")

    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        fetch_articles(db=db, progress_callback=progress_callback)
        logger.info("Article fetching completed successfully!")
    except Exception as e:
        logger.error(f"Error during article fetching: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def cmd_delete_old_articles(days: int = 90):
    """Delete articles older than specified number of days."""
    logger.info(f"Deleting articles older than {days} days...")

    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        deleted_count = delete_old_articles(db=db, days_old=days)
        logger.info(f"Successfully deleted {deleted_count} articles older than {days} days")
    except Exception as e:
        logger.error(f"Error during article deletion: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def cmd_update_profile_embeddings():
    """Calculate and update user profile embeddings."""
    logger.info("Calculating profile embeddings...")

    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        calculate_profile_embeddings(db=db)
        logger.info("Profile embeddings calculated successfully!")
    except Exception as e:
        logger.error(f"Error during profile embedding calculation: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def cmd_calculate_embedding_scores(days: int = 90):
    """Calculate embedding similarity scores for recent uninteracted articles."""
    logger.info(f"Calculating embedding scores for recent uninteracted articles (last {days} days)...")

    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        processed_count = calculate_scores_for_recent_uninteracted_articles(db=db, days=days)
        logger.info(f"Successfully calculated scores for {processed_count} articles")
    except Exception as e:
        logger.error(f"Error during embedding score calculation: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def cmd_recalculate_linucb_scores(top_n: int = 100):
    """Recalculate LinUCB scores for the top-N articles ranked by vector similarity."""
    logger.info(f"Recalculating LinUCB scores for top {top_n} articles...")

    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        updated_count = recalculate_scores(db=db, top_n=top_n, progress_callback=progress_callback)
        logger.info(f"Successfully updated LinUCB scores for {updated_count} articles")
    except Exception as e:
        logger.error(f"Error during LinUCB score recalculation: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def main():
    """Main command line interface."""
    parser = argparse.ArgumentParser(
        description="Service command utility for HabitFeed application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  create-db                    Create and initialize the database
  detect-topics               Run BERTopic topic detection on articles
  update-feeds                Update all RSS feeds
  fetch-articles              Fetch and process articles from feeds
  delete-old-articles         Delete articles older than specified days (--days)
  update-profile-embeddings   Calculate and update user profile embeddings
  calculate-embedding-scores  Calculate similarity scores for recent articles (--days)
  recalculate-linucb-scores   Recalculate LinUCB scores for top-N articles (--top-n)
        """
    )

    parser.add_argument('command', choices=[
        'create-db', 'detect-topics', 'update-feeds', 'fetch-articles',
        'delete-old-articles', 'update-profile-embeddings', 'calculate-embedding-scores', 'recalculate-linucb-scores'
    ], help='Command to execute')

    parser.add_argument('--days', type=int, default=90,
                       help='Number of days (for delete-old-articles and calculate-embedding-scores commands, default: 90)')

    parser.add_argument('--min-articles', type=int, default=5,
                       help='Minimum articles required for topic detection (default: 5)')

    parser.add_argument('--top-n', type=int, default=100,
                       help='Number of top articles for LinUCB score recalculation (default: 100)')

    args = parser.parse_args()

    try:
        if args.command == 'create-db':
            cmd_create_db()
        elif args.command == 'detect-topics':
            cmd_detect_topics(min_articles=args.min_articles)
        elif args.command == 'update-feeds':
            cmd_update_feeds()
        elif args.command == 'fetch-articles':
            cmd_fetch_articles()
        elif args.command == 'delete-old-articles':
            cmd_delete_old_articles(days=args.days)
        elif args.command == 'update-profile-embeddings':
            cmd_update_profile_embeddings()
        elif args.command == 'calculate-embedding-scores':
            cmd_calculate_embedding_scores(days=args.days)
        elif args.command == 'recalculate-linucb-scores':
            cmd_recalculate_linucb_scores(top_n=args.top_n)
        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
