import math
import numpy as np
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, text
from loguru import logger
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.models import Interaction, Article, Embedding, UserProfile, ActionEnum

#########################
# USER PROFILE EMBEDDINGS
#########################

# Constants for time decay calculation
HALF_LIFE_DAYS = 14
DECAY_LN2 = math.log(2)
LOOKBACK_DAYS = 90

# Negative preference weight factor for similarity scoring
NEGATIVE_WEIGHT_FACTOR = 0.7  # α factor for negative similarities

# Event weights as specified in scoring_and_matching.md
POSITIVE_WEIGHTS = {
    ActionEnum.click: 0.5,
    ActionEnum.upvote: 2,
    ActionEnum.save: 3,
}

NEGATIVE_WEIGHTS = {
    ActionEnum.meh: 1,
    ActionEnum.skip: 2,
    ActionEnum.downvote: 3,
}

def time_decay(age_days: float) -> float:
    """Calculate time decay factor using exponential decay with 14-day half-life."""
    return math.exp(-DECAY_LN2 * age_days / HALF_LIFE_DAYS)

def get_weighted_interactions(db: Session):
    """
    Query interactions from the last 90 days with their embeddings and calculate weights.
    Returns list of tuples: (embedding_blob, weight, is_positive)
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    
    # Query interactions with their article embeddings
    query = db.query(
        Interaction.action,
        Interaction.action_duration_ms,
        Interaction.event_ts,
        Embedding.vec
    ).join(
        Article, Interaction.article_id == Article.article_id
    ).join(
        Embedding, Article.article_id == Embedding.article_id
    ).filter(
        Interaction.event_ts >= cutoff_date
    )
    
    weighted_interactions = []
    
    for interaction in query.all():
        action, duration_ms, event_ts, embedding_blob = interaction
        
        # Calculate age in days
        # Ensure event_ts is timezone-aware (SQLite may return timezone-naive datetimes)
        if event_ts.tzinfo is None:
            event_ts = event_ts.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - event_ts).total_seconds() / (24 * 3600)
        
        # Calculate base weight and determine if positive/negative
        base_weight = 0
        is_positive = False
        
        if action in POSITIVE_WEIGHTS:
            base_weight = POSITIVE_WEIGHTS[action]
            is_positive = True
            
            # Add dwell bonus for long dwells (≥30 seconds)
            if action == ActionEnum.dwell and duration_ms and duration_ms >= 30000:
                base_weight += 1
                
        elif action in NEGATIVE_WEIGHTS:
            base_weight = NEGATIVE_WEIGHTS[action]
            is_positive = False
        else:
            # Handle dwell separately if it's not in positive weights
            if action == ActionEnum.dwell:
                if duration_ms and duration_ms >= 30000:
                    base_weight = 1
                    is_positive = True
                # Ignore short dwells
                else:
                    continue
            else:
                logger.warning(f"Unknown action type: {action}")
                continue
        
        # Apply time decay
        weight = base_weight * time_decay(age_days)
        
        weighted_interactions.append((embedding_blob, weight, is_positive))
    
    return weighted_interactions

def calculate_embedding_vectors(weighted_interactions):
    """
    Calculate positive and negative embedding vectors with their numerators and denominators.
    Returns: (pos_numerator, pos_denominator, neg_numerator, neg_denominator)
    """
    # Initialize vectors
    pos_numerator = np.zeros(768, dtype=np.float32)
    pos_denominator = 0.0
    neg_numerator = np.zeros(768, dtype=np.float32)
    neg_denominator = 0.0
    
    for embedding_blob, weight, is_positive in weighted_interactions:
        # Convert embedding blob to numpy array
        embedding_vector = np.frombuffer(embedding_blob, dtype=np.float32)
        
        if is_positive:
            pos_numerator += weight * embedding_vector
            pos_denominator += weight
        else:
            neg_numerator += weight * embedding_vector
            neg_denominator += weight
    
    return pos_numerator, pos_denominator, neg_numerator, neg_denominator

def calculate_profile_embeddings(db: Session):
    """
    Calculate and store user profile embeddings based on interactions.
    Calculates separate positive and negative embedding vectors using weighted interactions
    with time decay. Stores numerators and denominators separately for dynamic updates.
    """
    logger.info("Starting profile embedding calculation...")
    
    try:
        # Get weighted interactions
        weighted_interactions = get_weighted_interactions(db)
        logger.info(f"Found {len(weighted_interactions)} weighted interactions")
        
        if not weighted_interactions:
            logger.info("No interactions found, initializing empty profile")
            # Initialize with empty vectors for brand new user
            pos_numerator = np.zeros(768, dtype=np.float32)
            pos_denominator = 0.0
            neg_numerator = np.zeros(768, dtype=np.float32)
            neg_denominator = 0.0
        else:
            # Calculate embedding vectors
            pos_numerator, pos_denominator, neg_numerator, neg_denominator = calculate_embedding_vectors(weighted_interactions)
            logger.info(f"Positive profile: {pos_denominator:.3f} total weight")
            logger.info(f"Negative profile: {neg_denominator:.3f} total weight")
        
        # Get or create user profile (single user app)
        user_profile = db.query(UserProfile).first()
        if not user_profile:
            user_profile = UserProfile()
            db.add(user_profile)
        
        # Store the numerators and denominators as binary blobs
        user_profile.interest_vector_numerator = pos_numerator.tobytes()
        user_profile.interest_vector_denominator = np.array([pos_denominator], dtype=np.float32).tobytes()
        user_profile.negative_interest_vector_numerator = neg_numerator.tobytes()
        user_profile.negative_interest_vector_denominator = np.array([neg_denominator], dtype=np.float32).tobytes()
        user_profile.interest_vectors_updated_at = datetime.now(timezone.utc)
        
        db.commit()
        logger.info("Profile embeddings calculated and stored successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error calculating profile embeddings: {e}")
        db.rollback()
        return False

def get_current_profile_vectors(db: Session):
    """
    Retrieve and calculate current positive and negative profile vectors.
    Returns: (positive_vector, negative_vector) as normalized numpy arrays.
    Returns None vectors if no profile exists or denominators are zero.
    """
    user_profile = db.query(UserProfile).first()
    if not user_profile:
        return None, None
    
    try:
        # Load numerators and denominators
        pos_numerator = np.frombuffer(user_profile.interest_vector_numerator, dtype=np.float32)
        pos_denominator = np.frombuffer(user_profile.interest_vector_denominator, dtype=np.float32)[0]
        neg_numerator = np.frombuffer(user_profile.negative_interest_vector_numerator, dtype=np.float32)
        neg_denominator = np.frombuffer(user_profile.negative_interest_vector_denominator, dtype=np.float32)[0]
        
        # Calculate final vectors
        pos_vector = None
        if pos_denominator > 0:
            pos_vector = pos_numerator / pos_denominator
            # Optional L2 normalization for cosine distance stability
            pos_vector /= (np.linalg.norm(pos_vector) + 1e-9)
        
        neg_vector = None
        if neg_denominator > 0:
            neg_vector = neg_numerator / neg_denominator
            # Optional L2 normalization for cosine distance stability
            neg_vector /= (np.linalg.norm(neg_vector) + 1e-9)
        
        return pos_vector, neg_vector
        
    except Exception as e:
        logger.error(f"Error retrieving profile vectors: {e}")
        return None, None

#########################
# PROCESSING WEIGHTS FOR ARTICLES
#########################

#########################
# Helper functions
#########################

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    Returns similarity score between -1 and 1.
    """
    # Vectors should already be normalized, but add safety check
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def calculate_article_similarity_score(article_embedding: np.ndarray, pos_vector: np.ndarray, neg_vector: np.ndarray) -> float:
    """
    Calculate similarity score for an article based on positive and negative preference vectors.
    Formula: positive_similarity - α * negative_similarity
    
    Args:
        article_embedding: Article's embedding vector
        pos_vector: User's positive preference vector (can be None)
        neg_vector: User's negative preference vector (can be None)
    
    Returns:
        Similarity score (float between -1 and 1)
    """
    score = 0.0
    
    # Calculate positive similarity
    if pos_vector is not None:
        pos_similarity = cosine_similarity(article_embedding, pos_vector)
        score += pos_similarity
    
    # Subtract weighted negative similarity
    if neg_vector is not None:
        neg_similarity = cosine_similarity(article_embedding, neg_vector)
        score -= NEGATIVE_WEIGHT_FACTOR * neg_similarity
    
    # Clamp score to [-1, 1] range for safety
    return max(-1.0, min(1.0, score))

#########################
# CORE GENERATION OF ARTICLE SCORES
#########################

def calculate_article_embedding_weights(db: Session, article_ids: list = None):
    """
    Calculate and store vector similarity scores for articles based on user preference vectors.
    
    Args:
        db: Database session
        article_ids: Optional list of specific article IDs to process. If None, processes all articles with embeddings.
    
    Returns:
        Number of articles processed
    """
    logger.info("Starting article similarity score calculation...")
    
    try:
        # Get current user preference vectors
        pos_vector, neg_vector = get_current_profile_vectors(db)
        
        if pos_vector is None and neg_vector is None:
            logger.warning("No user preference vectors found. Cannot calculate similarity scores.")
            return 0
        
        # Query articles with embeddings
        query = db.query(Article, Embedding.vec).join(
            Embedding, Article.article_id == Embedding.article_id
        )
        
        # Filter by specific article IDs if provided
        if article_ids:
            query = query.filter(Article.article_id.in_(article_ids))
        
        articles_with_embeddings = query.all()
        
        if not articles_with_embeddings:
            logger.info("No articles with embeddings found.")
            return 0
        
        logger.info(f"Processing {len(articles_with_embeddings)} articles...")
        
        processed_count = 0
        
        for article, embedding_blob in articles_with_embeddings:
            try:
                # Convert embedding blob to numpy array
                article_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                
                # Calculate similarity score
                similarity_score = calculate_article_similarity_score(
                    article_embedding, pos_vector, neg_vector
                )
                
                # Store the score in the article
                article.vector_similarity_score = similarity_score
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing article {article.article_id}: {e}")
                continue
        
        # Commit all changes
        db.commit()
        logger.info(f"Successfully calculated similarity scores for {processed_count} articles")
        
        return processed_count
        
    except Exception as e:
        logger.error(f"Error calculating article similarity scores: {e}")
        db.rollback()
        return 0


def update_single_article_score(db: Session, article_id: int):
    """
    Update similarity score for a single article. Useful for real-time updates.
    
    Args:
        db: Database session
        article_id: ID of the article to update
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get user preference vectors
        pos_vector, neg_vector = get_current_profile_vectors(db)
        
        if pos_vector is None and neg_vector is None:
            logger.warning(f"No user preference vectors found for article {article_id}")
            return False
        
        # Get article and its embedding
        result = db.query(Article, Embedding.vec).join(
            Embedding, Article.article_id == Embedding.article_id
        ).filter(Article.article_id == article_id).first()
        
        if not result:
            logger.warning(f"Article {article_id} not found or has no embedding")
            return False
        
        article, embedding_blob = result
        
        # Calculate and store similarity score
        article_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        similarity_score = calculate_article_similarity_score(
            article_embedding, pos_vector, neg_vector
        )
        
        article.vector_similarity_score = similarity_score
        db.commit()
        
        logger.info(f"Updated similarity score for article {article_id}: {similarity_score:.4f}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating article {article_id} similarity score: {e}")
        db.rollback()
        return False


#########################
# INCREMENTAL PROFILE UPDATE FUNCTIONS
#########################

def calculate_interaction_weight(action: ActionEnum, duration_ms: int = None, event_ts: datetime = None) -> tuple[float, bool]:
    """
    Calculate weight and determine positivity for a single interaction.
    
    Args:
        action: The interaction action
        duration_ms: Duration in milliseconds (for dwell actions)
        event_ts: Timestamp of the interaction (defaults to now)
    
    Returns:
        Tuple of (weight, is_positive) where weight includes time decay
    """
    if event_ts is None:
        event_ts = datetime.now(timezone.utc)
    
    # Ensure event_ts is timezone-aware
    if event_ts.tzinfo is None:
        event_ts = event_ts.replace(tzinfo=timezone.utc)
    
    # Calculate age in days for time decay
    age_days = (datetime.now(timezone.utc) - event_ts).total_seconds() / (24 * 3600)
    
    # Calculate base weight and determine if positive/negative
    base_weight = 0
    is_positive = False
    
    if action in POSITIVE_WEIGHTS:
        base_weight = POSITIVE_WEIGHTS[action]
        is_positive = True
        
        # Add dwell bonus for long dwells (≥30 seconds)
        if action == ActionEnum.dwell and duration_ms and duration_ms >= 30000:
            base_weight += 1
            
    elif action in NEGATIVE_WEIGHTS:
        base_weight = NEGATIVE_WEIGHTS[action]
        is_positive = False
    else:
        # Handle dwell separately if it's not in positive weights
        if action == ActionEnum.dwell:
            if duration_ms and duration_ms >= 30000:
                base_weight = 1
                is_positive = True
            # Ignore short dwells
            else:
                return 0.0, False
        else:
            logger.warning(f"Unknown action type: {action}")
            return 0.0, False
    
    # Apply time decay
    weight = base_weight * time_decay(age_days)
    
    return weight, is_positive


def get_article_embedding_vector(db: Session, article_id: int) -> np.ndarray:
    """
    Get the embedding vector for a specific article.
    
    Args:
        db: Database session
        article_id: ID of the article
    
    Returns:
        Numpy array of the embedding vector, or None if not found
    """
    try:
        embedding_blob = db.query(Embedding.vec).filter(
            Embedding.article_id == article_id
        ).scalar()
        
        if embedding_blob is None:
            logger.warning(f"No embedding found for article {article_id}")
            return None
        
        return np.frombuffer(embedding_blob, dtype=np.float32)
        
    except Exception as e:
        logger.error(f"Error retrieving embedding for article {article_id}: {e}")
        return None


def update_profile_vectors_incremental(db: Session, article_id: int, action: ActionEnum, 
                                     duration_ms: int = None, event_ts: datetime = None) -> bool:
    """
    Incrementally update user profile vectors with a new interaction.
    
    This function:
    1. Calculates the weight for the new interaction
    2. Gets the article's embedding vector
    3. Updates the appropriate numerator by adding w * v
    4. Updates the appropriate denominator by adding w
    5. Saves the updated values to the database
    
    Args:
        db: Database session
        article_id: ID of the article that was interacted with
        action: The interaction action
        duration_ms: Duration in milliseconds (for dwell actions)
        event_ts: Timestamp of the interaction (defaults to now)
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Incrementally updating profile vectors for article {article_id}, action {action}")
    
    try:
        # Calculate weight for this interaction
        weight, is_positive = calculate_interaction_weight(action, duration_ms, event_ts)
        
        if weight == 0:
            logger.info(f"Interaction has zero weight, skipping profile update")
            return True
        
        # Get article embedding
        article_embedding = get_article_embedding_vector(db, article_id)
        if article_embedding is None:
            logger.warning(f"Cannot update profile: no embedding for article {article_id}")
            return False
        
        # Get or create user profile
        user_profile = db.query(UserProfile).first()
        if not user_profile:
            logger.info("No user profile exists, creating new one")
            user_profile = UserProfile()
            # Initialize with zero vectors
            user_profile.interest_vector_numerator = np.zeros(768, dtype=np.float32).tobytes()
            user_profile.interest_vector_denominator = np.array([0.0], dtype=np.float32).tobytes()
            user_profile.negative_interest_vector_numerator = np.zeros(768, dtype=np.float32).tobytes()
            user_profile.negative_interest_vector_denominator = np.array([0.0], dtype=np.float32).tobytes()
            db.add(user_profile)
        
        # Load current numerators and denominators
        pos_numerator = np.frombuffer(user_profile.interest_vector_numerator, dtype=np.float32).copy()
        pos_denominator = np.frombuffer(user_profile.interest_vector_denominator, dtype=np.float32)[0]
        neg_numerator = np.frombuffer(user_profile.negative_interest_vector_numerator, dtype=np.float32).copy()
        neg_denominator = np.frombuffer(user_profile.negative_interest_vector_denominator, dtype=np.float32)[0]
        
        # Update appropriate vectors
        if is_positive:
            pos_numerator += weight * article_embedding
            pos_denominator += weight
            logger.info(f"Updated positive profile: added weight {weight:.4f}, new total weight {pos_denominator:.4f}")
        else:
            neg_numerator += weight * article_embedding
            neg_denominator += weight
            logger.info(f"Updated negative profile: added weight {weight:.4f}, new total weight {neg_denominator:.4f}")
        
        # Store updated values back to database
        user_profile.interest_vector_numerator = pos_numerator.tobytes()
        user_profile.interest_vector_denominator = np.array([pos_denominator], dtype=np.float32).tobytes()
        user_profile.negative_interest_vector_numerator = neg_numerator.tobytes()
        user_profile.negative_interest_vector_denominator = np.array([neg_denominator], dtype=np.float32).tobytes()
        user_profile.interest_vectors_updated_at = datetime.now(timezone.utc)
        
        db.commit()
        logger.info("Profile vectors updated incrementally")
        return True
        
    except Exception as e:
        logger.error(f"Error updating profile vectors incrementally: {e}")
        db.rollback()
        return False


def update_profile_and_article_score_on_interaction(db: Session, article_id: int, action: ActionEnum,
                                                   duration_ms: int = None, event_ts: datetime = None) -> bool:
    """
    Convenience function that updates both the user profile vectors and the article's similarity score
    when a new interaction occurs.
    
    This function:
    1. Updates the user profile vectors incrementally
    2. Recalculates the similarity score for the interacted article (sets to -2 for explicit interactions)
    3. Optionally recalculates scores for other recent uninteracted articles if desired
    
    Args:
        db: Database session
        article_id: ID of the article that was interacted with
        action: The interaction action
        duration_ms: Duration in milliseconds (for dwell actions)
        event_ts: Timestamp of the interaction (defaults to now)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Update profile vectors incrementally
        profile_updated = update_profile_vectors_incremental(db, article_id, action, duration_ms, event_ts)
        
        if not profile_updated:
            logger.error("Failed to update profile vectors")
            return False
        
        # For explicit interactions, mark article with low score to avoid re-recommendation
        explicit_actions = [
            ActionEnum.click, ActionEnum.upvote, ActionEnum.downvote,
            ActionEnum.save, ActionEnum.meh, ActionEnum.skip
        ]
        
        if action in explicit_actions:
            # Set vector_similarity_score to -2 for this article
            article = db.query(Article).filter(Article.article_id == article_id).first()
            if article:
                article.vector_similarity_score = -2.0
                logger.info(f"Marked article {article_id} with low score (-2.0) due to explicit interaction")
        else:
            # For non-explicit interactions (like dwell), recalculate the article's score with updated profile
            update_single_article_score(db, article_id)
        
        db.commit()
        logger.info(f"Successfully processed interaction for article {article_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing interaction for article {article_id}: {e}")
        db.rollback()
        return False

#########################
# CONVENIENCE FUNCTIONS
#########################

def calculate_scores_for_recent_uninteracted_articles(db: Session, days: int = LOOKBACK_DAYS):
    """
    Calculate similarity scores for recent articles that the user hasn't explicitly interacted with.
    
    This function:
    1. Gets articles with pub_date within specified days window
    2. Excludes articles that have 'click', 'upvote', 'downvote', 'save', 'meh', or 'skip' 
       interactions within specified days window
    3. Calculates similarity scores for the remaining articles
    
    Args:
        db: Database session
        days: Number of days to look back for recent articles (defaults to LOOKBACK_DAYS)
    
    Returns:
        Number of articles processed
    """
    logger.info(f"Calculating scores for recent uninteracted articles (last {days} days)...")
    
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Define explicit interaction types we want to exclude
        explicit_actions = [
            ActionEnum.click,
            ActionEnum.upvote, 
            ActionEnum.downvote,
            ActionEnum.save,
            ActionEnum.meh,
            ActionEnum.skip
        ]
        
        # Get articles with pub_date within specified days window
        recent_articles_query = db.query(Article.article_id).filter(
            Article.pub_date >= cutoff_date
        )
        
        # Get articles that have explicit interactions within specified days window
        interacted_articles_query = db.query(Interaction.article_id.distinct()).filter(
            and_(
                Interaction.event_ts >= cutoff_date,
                Interaction.action.in_(explicit_actions)
            )
        )
        
        # Get article IDs that are recent but not explicitly interacted with
        uninteracted_article_ids = recent_articles_query.filter(
            ~Article.article_id.in_(interacted_articles_query)
        ).all()
        
        article_ids = [row.article_id for row in uninteracted_article_ids]
        
        if not article_ids:
            logger.info("No recent uninteracted articles found")
            return 0
        
        logger.info(f"Found {len(article_ids)} recent articles without explicit interactions")
        
        # Calculate similarity scores for these articles
        processed_count = calculate_article_embedding_weights(db, article_ids=article_ids)
        
        logger.info(f"Calculated scores for {processed_count} recent uninteracted articles")
        return processed_count
        
    except Exception as e:
        logger.error(f"Error calculating scores for recent uninteracted articles: {e}")
        return 0


def mark_interacted_articles_low_score(db: Session):
    """
    Set vector_similarity_score to -2 for all articles the user has ever had explicit interactions with.
    
    This marks articles with 'click', 'upvote', 'downvote', 'save', 'meh', or 'skip' interactions
    as low priority for recommendations since the user has already engaged with them.
    
    Args:
        db: Database session
    
    Returns:
        Number of articles marked with low scores
    """
    logger.info("Marking all previously interacted articles with low scores...")
    
    try:
        # Define explicit interaction types
        explicit_actions = [
            ActionEnum.click,
            ActionEnum.upvote,
            ActionEnum.downvote, 
            ActionEnum.save,
            ActionEnum.meh,
            ActionEnum.skip
        ]
        
        # Get all articles that have ever had explicit interactions
        interacted_article_ids = db.query(Interaction.article_id.distinct()).filter(
            Interaction.action.in_(explicit_actions)
        ).all()
        
        if not interacted_article_ids:
            logger.info("No articles with explicit interactions found")
            return 0
        
        # Extract article_id values from the query result
        article_ids = [row[0] for row in interacted_article_ids]
        
        logger.info(f"Found {len(article_ids)} articles with explicit interactions")
        
        # Update vector_similarity_score to -2 for these articles
        updated_count = db.query(Article).filter(
            Article.article_id.in_(article_ids)
        ).update(
            {"vector_similarity_score": -2.0},
            synchronize_session=False
        )
        
        db.commit()
        
        logger.info(f"Marked {updated_count} previously interacted articles with low scores (-2.0)")
        return updated_count
        
    except Exception as e:
        logger.error(f"Error marking interacted articles with low scores: {e}")
        db.rollback()
        return 0 

if __name__ == "__main__":
    from db.database import SessionLocal
    db = SessionLocal()
    # if profile stable, this isn't needed as much
    ## Note, half-life of 14 days means that there is a time decay on the profile
    calculate_profile_embeddings(db)

    # This is needed to run regularly to actually calculate scores for articles
    calculate_scores_for_recent_uninteracted_articles(db)
    # The below shouldn't be needed to run regularly since every interaction
    # should set the score to -2.0 anyways
    # but just in case, it's here
    # mark_interacted_articles_low_score(db)



## USING CONVENIENCE FUNCTIONS FOR INCREMENTAL PROFILE UPDATE
""" success = update_profile_vectors_incremental(
    db, 
    article_id=123, 
    action=ActionEnum.upvote,
    event_ts=datetime.now(timezone.utc)
)

# Or use the full convenience function that also handles article scoring
success = update_profile_and_article_score_on_interaction(
    db,
    article_id=123,
    action=ActionEnum.upvote
) """