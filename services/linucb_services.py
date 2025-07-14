import numpy as np
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, text, func
from loguru import logger
from typing import List, Optional, Tuple
import sys
import os
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.models import Article, Embedding, LinucbState, ActionEnum, Interaction, Topic, article_topic_association, Feed

# LinUCB hyperparameters
ALPHA = 1.0  # Exploration parameter - higher values encourage more exploration
FEATURE_DIM = 16  # Dimension of engineered feature vector
DEFAULT_USER_ID = 1  # Single user system for now

# Reward mapping for different actions
ACTION_REWARDS = {
    ActionEnum.click: 0.5,
    ActionEnum.upvote: 1.0,
    ActionEnum.save: 1.0,
    ActionEnum.dwell: 0.7,  # Will adjust based on duration
    ActionEnum.meh: -0.3,
    ActionEnum.skip: -0.5,
    ActionEnum.downvote: -1.0,
}

def get_action_reward(action: ActionEnum, duration_ms: Optional[int] = None) -> float:
    """
    Convert user action to reward signal for LinUCB.
    
    Args:
        action: The user action taken
        duration_ms: Duration for dwell actions
        
    Returns:
        Reward value between -1.0 and 1.0
    """
    base_reward = ACTION_REWARDS.get(action, 0.0)
    
    # Adjust dwell reward based on duration
    if action == ActionEnum.dwell and duration_ms is not None:
        if duration_ms >= 30000:  # 30+ seconds
            base_reward = 1.0
        elif duration_ms >= 10000:  # 10-30 seconds  
            base_reward = 0.5
        elif duration_ms >= 3000:  # 3-10 seconds
            base_reward = 0.2
        else:  # < 3 seconds
            base_reward = -0.1
    
    return base_reward

def extract_article_features(db: Session, article_id: int) -> Optional[np.ndarray]:
    """
    Extract engineered features for an article to use as LinUCB context vector.
    
    Features (16 dimensions):
    0. vector_similarity_score (normalized)
    1-3. Top 3 topic confidences
    4. feed_id (normalized by total feeds)
    5. recency_score (days since publication, normalized)
    6. hour_of_day (normalized 0-1)
    7. day_of_week (normalized 0-1) 
    8. article_length_bucket (short/medium/long: 0/0.5/1)
    9. feed_interaction_rate (user's interaction rate with this feed)
    10. topic_diversity (entropy of topic distribution)
    11. reading_time_estimate (normalized)
    12. similarity_to_recent_positives (avg similarity to recent positive articles)
    13. time_since_last_feed_interaction (normalized)
    14. weekend_indicator (0 for weekday, 1 for weekend)
    15. bias term (always 1.0)
    
    Args:
        db: Database session
        article_id: Article ID
        
    Returns:
        Feature vector as numpy array, or None if article not found
    """
    try:
        # Get article with feed and topics
        article = db.query(Article).filter(Article.article_id == article_id).first()
        if not article:
            logger.warning(f"Article {article_id} not found")
            return None
            
        features = np.zeros(FEATURE_DIM, dtype=np.float32)
        
        # Feature 0: vector_similarity_score (clamp and normalize to [0,1])
        if article.vector_similarity_score is not None:
            # Clamp to [-1, 1] then shift to [0, 1]
            clamped_score = max(-1.0, min(1.0, article.vector_similarity_score))
            features[0] = (clamped_score + 1.0) / 2.0
        else:
            features[0] = 0.5  # neutral value
            
        # Features 1-3: Top 3 topic confidences
        topic_confidences = db.query(article_topic_association.c.confidence).filter(
            article_topic_association.c.article_id == article_id
        ).order_by(article_topic_association.c.confidence.desc()).limit(3).all()
        
        for i, (confidence,) in enumerate(topic_confidences):
            if i < 3:
                features[1 + i] = min(1.0, max(0.0, confidence))  # Clamp to [0,1]
        
        # Feature 4: feed_id (normalized by total number of feeds)
        total_feeds = db.query(func.count(Feed.feed_id)).scalar() or 1
        if article.feed_id:
            features[4] = article.feed_id / total_feeds
            
        # Features 5-7: Temporal features
        if article.pub_date:
            now = datetime.now(timezone.utc)
            if article.pub_date.tzinfo is None:
                pub_date = article.pub_date.replace(tzinfo=timezone.utc)
            else:
                pub_date = article.pub_date
                
            # Feature 5: Recency (days since publication, capped at 30 days)
            days_old = (now - pub_date).total_seconds() / (24 * 3600)
            features[5] = max(0.0, min(1.0, 1.0 - (days_old / 30.0)))  # 1.0 = very recent, 0.0 = 30+ days old
            
            # Feature 6: Hour of day (0-1)
            features[6] = pub_date.hour / 23.0
            
            # Feature 7: Day of week (0-1)
            features[7] = pub_date.weekday() / 6.0
            
            # Feature 14: Weekend indicator
            features[14] = 1.0 if pub_date.weekday() >= 5 else 0.0
        else:
            features[5] = 0.0  # Unknown publication date
            features[6] = 0.5  # Neutral hour
            features[7] = 0.5  # Neutral day
            
        # Feature 8: Article length bucket
        if article.full_text:
            text_length = len(article.full_text)
            if text_length < 1000:
                features[8] = 0.0  # Short
            elif text_length < 5000:
                features[8] = 0.5  # Medium
            else:
                features[8] = 1.0  # Long
        else:
            features[8] = 0.0
            
        # Feature 9: Feed interaction rate (interactions per article for this feed)
        if article.feed_id:
            feed_article_count = db.query(func.count(Article.article_id)).filter(
                Article.feed_id == article.feed_id
            ).scalar() or 1
            
            feed_interaction_count = db.query(func.count(Interaction.int_id)).join(
                Article, Interaction.article_id == Article.article_id
            ).filter(Article.feed_id == article.feed_id).scalar() or 0
            
            features[9] = min(1.0, feed_interaction_count / feed_article_count)
        else:
            features[9] = 0.0
            
        # Feature 10: Topic diversity (entropy of topic distribution)
        all_confidences = [conf[0] for conf in topic_confidences]
        if all_confidences:
            # Normalize confidences to sum to 1
            total_conf = sum(all_confidences)
            if total_conf > 0:
                normalized_conf = [c / total_conf for c in all_confidences]
                # Calculate entropy
                entropy = -sum(p * math.log(p + 1e-10) for p in normalized_conf if p > 0)
                max_entropy = math.log(len(normalized_conf)) if len(normalized_conf) > 1 else 1.0
                features[10] = entropy / max_entropy if max_entropy > 0 else 0.0
            else:
                features[10] = 0.0
        else:
            features[10] = 0.0
            
        # Feature 11: Reading time estimate (words per minute)
        if article.full_text:
            word_count = len(article.full_text.split())
            reading_time_minutes = word_count / 200  # Assume 200 WPM
            features[11] = min(1.0, reading_time_minutes / 30.0)  # Normalize to 30-minute max
        else:
            features[11] = 0.0
            
        # Feature 12: Similarity to recent positive articles
        # Get recent positive interactions (last 30 days)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
        positive_actions = [ActionEnum.upvote, ActionEnum.save, ActionEnum.click]
        
        recent_positive_articles = db.query(Article.article_id).join(
            Interaction, Article.article_id == Interaction.article_id
        ).filter(
            and_(
                Interaction.event_ts >= cutoff_date,
                Interaction.action.in_(positive_actions)
            )
        ).distinct().limit(10).all()  # Limit to recent 10 for performance
        
        if recent_positive_articles and article.vector_similarity_score is not None:
            # Simple heuristic: if current article has high similarity score, 
            # it's likely similar to recent positives
            features[12] = max(0.0, min(1.0, (article.vector_similarity_score + 1.0) / 2.0))
        else:
            features[12] = 0.0
            
        # Feature 13: Time since last interaction with this feed
        if article.feed_id:
            last_interaction = db.query(func.max(Interaction.event_ts)).join(
                Article, Interaction.article_id == Article.article_id
            ).filter(Article.feed_id == article.feed_id).scalar()
            
            if last_interaction:
                if last_interaction.tzinfo is None:
                    last_interaction = last_interaction.replace(tzinfo=timezone.utc)
                days_since = (datetime.now(timezone.utc) - last_interaction).total_seconds() / (24 * 3600)
                features[13] = min(1.0, days_since / 7.0)  # Normalize to 7 days
            else:
                features[13] = 1.0  # No interactions = long time
        else:
            features[13] = 1.0
            
        # Feature 15: Bias term
        features[15] = 1.0
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features for article {article_id}: {e}")
        return None

def get_or_create_linucb_state(db: Session, user_id: int = DEFAULT_USER_ID) -> LinucbState:
    """
    Get existing LinUCB state for user or create a new one with identity matrix initialization.
    
    Args:
        db: Database session
        user_id: User ID (default 1 for single user system)
        
    Returns:
        LinucbState object
    """
    state = db.query(LinucbState).filter(LinucbState.user_id == user_id).first()
    
    if not state:
        logger.info(f"Creating new LinUCB state for user {user_id}")
        
        # Initialize A as identity matrix and b as zero vector
        A_matrix = np.eye(FEATURE_DIM, dtype=np.float32)
        b_vector = np.zeros(FEATURE_DIM, dtype=np.float32)
        
        state = LinucbState(
            user_id=user_id,
            A=A_matrix.tobytes(),
            b=b_vector.tobytes(),
            last_update=datetime.now(timezone.utc)
        )
        db.add(state)
        db.commit()
        
    return state

def load_state_matrices(state: LinucbState) -> tuple[np.ndarray, np.ndarray]:
    """
    Load A matrix and b vector from LinucbState object.
    
    Args:
        state: LinucbState database object
        
    Returns:
        Tuple of (A_matrix, b_vector) as numpy arrays
    """
    A_matrix = np.frombuffer(state.A, dtype=np.float32).reshape(FEATURE_DIM, FEATURE_DIM).copy()
    b_vector = np.frombuffer(state.b, dtype=np.float32).copy()
    return A_matrix, b_vector

def save_state_matrices(state: LinucbState, A_matrix: np.ndarray, b_vector: np.ndarray, db: Session):
    """
    Save A matrix and b vector back to LinucbState object.
    
    Args:
        state: LinucbState database object
        A_matrix: Updated A matrix
        b_vector: Updated b vector
        db: Database session
    """
    state.A = A_matrix.astype(np.float32).tobytes()
    state.b = b_vector.astype(np.float32).tobytes()
    state.last_update = datetime.now(timezone.utc)
    db.add(state)

def get_article_features(db: Session, article_id: int) -> Optional[np.ndarray]:
    """
    Get the feature vector for an article.
    
    Args:
        db: Database session
        article_id: Article ID
        
    Returns:
        Feature vector as numpy array, or None if extraction failed
    """
    return extract_article_features(db, article_id)

def update_model(db: Session, article_id: int, action: ActionEnum, 
                duration_ms: Optional[int] = None, user_id: int = DEFAULT_USER_ID) -> bool:
    """
    Update the LinUCB model based on user interaction with an article.
    
    This implements the LinUCB update rule:
    - A_t+1 = A_t + x_t * x_t^T
    - b_t+1 = b_t + r_t * x_t
    
    Args:
        db: Database session
        article_id: ID of the article that was interacted with
        action: The action taken by the user
        duration_ms: Duration of interaction (for dwell actions)
        user_id: User ID (default 1 for single user system)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get article features (context vector)
        x_t = get_article_features(db, article_id)
        if x_t is None:
            logger.error(f"Cannot update LinUCB model: no features for article {article_id}")
            return False
        
        # Get reward signal
        r_t = get_action_reward(action, duration_ms)
        logger.info(f"LinUCB update: article {article_id}, action {action}, reward {r_t}")
        
        # Get or create LinUCB state
        state = get_or_create_linucb_state(db, user_id)
        A_matrix, b_vector = load_state_matrices(state)
        
        # Update A matrix: A_t+1 = A_t + x_t * x_t^T
        A_matrix += np.outer(x_t, x_t)
        
        # Update b vector: b_t+1 = b_t + r_t * x_t  
        b_vector += r_t * x_t
        
        # Save updated state
        save_state_matrices(state, A_matrix, b_vector, db)
        db.commit()
        
        logger.info(f"LinUCB model updated successfully for user {user_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating LinUCB model: {e}")
        db.rollback()
        return False

def calculate_linucb_score(x: np.ndarray, A_matrix: np.ndarray, b_vector: np.ndarray, alpha: float = ALPHA) -> float:
    """
    Calculate LinUCB upper confidence bound score for a context vector.
    
    The LinUCB score is: theta^T * x + alpha * sqrt(x^T * A^-1 * x)
    where theta = A^-1 * b
    
    Args:
        x: Context vector (article features)
        A_matrix: A matrix from LinUCB state
        b_vector: b vector from LinUCB state
        alpha: Exploration parameter
        
    Returns:
        LinUCB score (higher = more likely to be selected)
    """
    try:
        # Calculate A^-1
        A_inv = np.linalg.inv(A_matrix)
        
        # Calculate theta = A^-1 * b
        theta = A_inv @ b_vector
        
        # Calculate confidence width: alpha * sqrt(x^T * A^-1 * x)
        confidence_width = alpha * np.sqrt(x.T @ A_inv @ x)
        
        # LinUCB score: theta^T * x + confidence_width
        score = theta.T @ x + confidence_width
        
        return float(score)
        
    except np.linalg.LinAlgError:
        logger.warning("LinUCB A matrix is singular, using exploration-only score")
        # If A is singular, fall back to exploration based on uncertainty
        return alpha * np.random.random()

def recalculate_scores(db: Session, top_n: int = 100, user_id: int = DEFAULT_USER_ID, progress_callback=None) -> int:
    """
    Recalculate LinUCB scores for the top-N articles ranked by vector similarity.
    
    Args:
        db: Database session
        top_n: Number of top articles to calculate LinUCB scores for
        user_id: User ID (default 1 for single user system)
        progress_callback: Optional callback function(current, total, message) for progress updates
        
    Returns:
        Number of articles that had their LinUCB scores updated
    """
    try:
        logger.info(f"Recalculating LinUCB scores for top {top_n} articles")
        
        if progress_callback:
            progress_callback(0, top_n, "Starting LinUCB score recalculation...")
        
        # Get LinUCB state
        state = get_or_create_linucb_state(db, user_id)
        A_matrix, b_vector = load_state_matrices(state)
        
        if progress_callback:
            progress_callback(0, top_n, "Loaded LinUCB state, fetching articles...")
        
        # Get top-N articles by vector similarity score
        top_articles = db.query(Article).filter(
            Article.vector_similarity_score.isnot(None)
        ).order_by(Article.vector_similarity_score.desc()).limit(top_n).all()
        
        total_articles = len(top_articles)
        updated_count = 0
        
        if progress_callback:
            progress_callback(0, total_articles, f"Processing {total_articles} articles...")
        
        for i, article in enumerate(top_articles):
            # Get article features
            x = get_article_features(db, article.article_id)
            if x is None:
                if progress_callback:
                    progress_callback(i + 1, total_articles, f"Skipped article {article.article_id} (no features)")
                continue
                
            # Calculate LinUCB score
            linucb_score = calculate_linucb_score(x, A_matrix, b_vector)
            
            # Update article with LinUCB score
            article.linucb_score = linucb_score
            db.add(article)
            updated_count += 1
            
            if progress_callback:
                progress_callback(i + 1, total_articles, f"Updated article {article.article_id} (score: {linucb_score:.3f})")
        
        db.commit()
        logger.info(f"Updated LinUCB scores for {updated_count} articles")
        
        if progress_callback:
            progress_callback(total_articles, total_articles, f"Completed! Updated {updated_count} articles")
        
        return updated_count
        
    except Exception as e:
        logger.error(f"Error recalculating LinUCB scores: {e}")
        if progress_callback:
            progress_callback(0, top_n, f"Error: {str(e)}")
        db.rollback()
        return 0



def reset_linucb_state(db: Session, user_id: int = DEFAULT_USER_ID) -> bool:
    """
    Reset LinUCB state to initial values (identity matrix for A, zero vector for b).
    Useful for testing or if the model needs to be restarted.
    
    Args:
        db: Database session
        user_id: User ID to reset
        
    Returns:
        True if successful, False otherwise
    """
    try:
        state = db.query(LinucbState).filter(LinucbState.user_id == user_id).first()
        if state:
            A_matrix = np.eye(FEATURE_DIM, dtype=np.float32)
            b_vector = np.zeros(FEATURE_DIM, dtype=np.float32)
            save_state_matrices(state, A_matrix, b_vector, db)
            db.commit()
            logger.info(f"Reset LinUCB state for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error resetting LinUCB state: {e}")
        db.rollback()
        return False

def get_feature_names() -> List[str]:
    """
    Get human-readable names for each feature dimension.
    Useful for debugging and understanding the feature vector.
    
    Returns:
        List of feature names corresponding to each dimension
    """
    return [
        "vector_similarity_score",       # 0
        "topic_1_confidence",            # 1
        "topic_2_confidence",            # 2 
        "topic_3_confidence",            # 3
        "feed_id_normalized",            # 4
        "recency_score",                 # 5
        "hour_of_day",                   # 6
        "day_of_week",                   # 7
        "article_length_bucket",         # 8
        "feed_interaction_rate",         # 9
        "topic_diversity",               # 10
        "reading_time_estimate",         # 11
        "similarity_to_recent_positives", # 12
        "time_since_last_feed_interaction", # 13
        "weekend_indicator",             # 14
        "bias_term"                      # 15
    ]

def explain_article_features(db: Session, article_id: int) -> Optional[dict]:
    """
    Extract and explain features for an article for debugging/understanding.
    
    Args:
        db: Database session
        article_id: Article ID
        
    Returns:
        Dictionary mapping feature names to values, or None if extraction failed
    """
    features = extract_article_features(db, article_id)
    if features is None:
        return None
        
    feature_names = get_feature_names()
    return {name: float(value) for name, value in zip(feature_names, features)}