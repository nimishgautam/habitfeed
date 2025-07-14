import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from db.models import Article, Feed, Embedding, Interaction, UserProfile, ActionEnum
from services.embedding_services import (
    calculate_profile_embeddings,
    calculate_article_embedding_weights,
    calculate_article_similarity_score,
    get_current_profile_vectors,
    update_single_article_score,
    calculate_scores_for_recent_uninteracted_articles,
    mark_interacted_articles_low_score,
    time_decay,
    cosine_similarity,
    POSITIVE_WEIGHTS,
    NEGATIVE_WEIGHTS,
    NEGATIVE_WEIGHT_FACTOR,
    LOOKBACK_DAYS
)
from tests.dummy_articles import DUMMY_ARTICLES


def create_mock_embedding(category: str, article_key: str) -> np.ndarray:
    """
    Create mock embeddings that are similar within categories but different across categories.
    This simulates how real embeddings would cluster similar content together.
    """
    # Base vectors for different categories
    category_bases = {
        'tech': np.array([1.0, 0.8, 0.2, 0.1] + [0.0] * 764),
        'health': np.array([0.1, 0.2, 1.0, 0.8] + [0.0] * 764),
        'environment': np.array([0.2, 1.0, 0.1, 0.3] + [0.0] * 764),
        'finance': np.array([0.8, 0.1, 0.3, 1.0] + [0.0] * 764)
    }
    
    # Add some variation within category based on article key
    variation = hash(article_key) % 100 / 1000.0  # Small variation 0-0.099
    base_vector = category_bases[category].copy()
    base_vector[0] += variation
    base_vector[1] -= variation * 0.5
    
    # Normalize to unit vector (typical for embeddings)
    norm = np.linalg.norm(base_vector)
    if norm > 0:
        base_vector = base_vector / norm
    
    return base_vector.astype(np.float32)


@pytest.fixture
def sample_feed(db_session):
    """Create a sample feed for testing."""
    import uuid
    # Use a unique URL for each test to avoid constraint violations
    unique_id = str(uuid.uuid4())[:8]
    feed = Feed(
        feed_url=f"https://test-{unique_id}.example.com/feed.xml",
        title="Test Feed",
        last_checked=datetime.now(timezone.utc)
    )
    db_session.add(feed)
    db_session.commit()
    db_session.refresh(feed)
    return feed


@pytest.fixture
def articles_with_embeddings(db_session, sample_feed):
    """Create articles with mock embeddings for testing."""
    articles = {}
    
    for article_key, article_data in DUMMY_ARTICLES.items():
        # Create article
        article = Article(
            feed_id=sample_feed.feed_id,
            guid=f"guid_{article_key}",
            url=article_data["url"],
            title=article_data["title"],
            rss_description=article_data["description"],
            full_text=article_data["content"],
            pub_date=datetime.now(timezone.utc) - timedelta(days=5),  # Recent articles
            language="en"
        )
        db_session.add(article)
        db_session.commit()
        db_session.refresh(article)
        
        # Create mock embedding based on category
        category = article_key.split('_')[0]  # 'tech', 'health', etc.
        mock_embedding = create_mock_embedding(category, article_key)
        
        embedding = Embedding(
            article_id=article.article_id,
            vec=mock_embedding.tobytes()
        )
        db_session.add(embedding)
        
        articles[article_key] = article
    
    db_session.commit()
    return articles


@pytest.fixture
def interactions_setup(db_session, articles_with_embeddings):
    """Create a realistic set of interactions for testing recommendation logic."""
    # Positive interactions with tech articles (user likes tech)
    tech_ai_article = articles_with_embeddings['tech_ai']
    tech_quantum_article = articles_with_embeddings['tech_quantum']
    
    # Strong positive signals for tech articles
    interactions = [
        Interaction(
            article_id=tech_ai_article.article_id,
            action=ActionEnum.click,
            event_ts=datetime.now(timezone.utc) - timedelta(days=2)
        ),
        Interaction(
            article_id=tech_ai_article.article_id,
            action=ActionEnum.upvote,
            event_ts=datetime.now(timezone.utc) - timedelta(days=2)
        ),
        Interaction(
            article_id=tech_ai_article.article_id,
            action=ActionEnum.save,
            event_ts=datetime.now(timezone.utc) - timedelta(days=1)
        ),
        Interaction(
            article_id=tech_quantum_article.article_id,
            action=ActionEnum.click,
            event_ts=datetime.now(timezone.utc) - timedelta(days=3)
        ),
        Interaction(
            article_id=tech_quantum_article.article_id,
            action=ActionEnum.upvote,
            event_ts=datetime.now(timezone.utc) - timedelta(days=3)
        ),
    ]
    
    # Negative interactions with finance articles (user dislikes finance)
    finance_markets_article = articles_with_embeddings['finance_markets']
    finance_crypto_article = articles_with_embeddings['finance_crypto']
    
    interactions.extend([
        Interaction(
            article_id=finance_markets_article.article_id,
            action=ActionEnum.click,
            event_ts=datetime.now(timezone.utc) - timedelta(days=4)
        ),
        Interaction(
            article_id=finance_markets_article.article_id,
            action=ActionEnum.meh,
            event_ts=datetime.now(timezone.utc) - timedelta(days=4)
        ),
        Interaction(
            article_id=finance_crypto_article.article_id,
            action=ActionEnum.skip,
            event_ts=datetime.now(timezone.utc) - timedelta(days=5)
        ),
        Interaction(
            article_id=finance_crypto_article.article_id,
            action=ActionEnum.downvote,
            event_ts=datetime.now(timezone.utc) - timedelta(days=5)
        ),
    ])
    
    # Add some dwell interactions
    health_nutrition_article = articles_with_embeddings['health_nutrition']
    interactions.extend([
        # Short dwell (should be ignored)
        Interaction(
            article_id=health_nutrition_article.article_id,
            action=ActionEnum.dwell,
            action_duration_ms=15000,  # 15 seconds
            event_ts=datetime.now(timezone.utc) - timedelta(days=3)
        ),
        # Long dwell (should count as positive)
        Interaction(
            article_id=health_nutrition_article.article_id,
            action=ActionEnum.dwell,
            action_duration_ms=45000,  # 45 seconds
            event_ts=datetime.now(timezone.utc) - timedelta(days=2)
        ),
    ])
    
    for interaction in interactions:
        db_session.add(interaction)
    
    db_session.commit()
    return interactions


class TestEmbeddingCalculations:
    """Test core embedding calculation functions."""
    
    def test_time_decay(self):
        """Test time decay function with known values."""
        # At 0 days, decay should be 1.0
        assert time_decay(0) == 1.0
        
        # At 14 days (half-life), decay should be 0.5
        assert abs(time_decay(14) - 0.5) < 0.001
        
        # At 28 days (two half-lives), decay should be 0.25
        assert abs(time_decay(28) - 0.25) < 0.001
        
        # Older events should have lower weights
        assert time_decay(7) > time_decay(14)
        assert time_decay(14) > time_decay(21)
    
    def test_cosine_similarity(self):
        """Test cosine similarity function."""
        # Identical vectors should have similarity 1.0
        vec_a = np.array([1.0, 2.0, 3.0])
        vec_b = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(vec_a, vec_b) - 1.0) < 0.001
        
        # Orthogonal vectors should have similarity 0.0
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.0, 1.0, 0.0])
        assert abs(cosine_similarity(vec_a, vec_b)) < 0.001
        
        # Opposite vectors should have similarity -1.0
        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([-1.0, 0.0, 0.0])
        assert abs(cosine_similarity(vec_a, vec_b) - (-1.0)) < 0.001
        
        # Zero vectors should return 0.0
        vec_a = np.array([0.0, 0.0, 0.0])
        vec_b = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(vec_a, vec_b) == 0.0


class TestProfileEmbeddings:
    """Test user profile embedding calculation and retrieval."""
    
    def test_calculate_profile_embeddings_no_interactions(self, db_session):
        """Test profile calculation with no interactions (new user)."""
        result = calculate_profile_embeddings(db_session)
        assert result is True
        
        # Should create a user profile with zero vectors
        user_profile = db_session.query(UserProfile).first()
        assert user_profile is not None
        assert user_profile.interest_vectors_updated_at is not None
        
        # Check that vectors are initialized to zero
        pos_vector, neg_vector = get_current_profile_vectors(db_session)
        assert pos_vector is None  # No positive interactions
        assert neg_vector is None  # No negative interactions
    
    def test_calculate_profile_embeddings_with_interactions(self, db_session, articles_with_embeddings, interactions_setup):
        """Test profile calculation with realistic interactions."""
        result = calculate_profile_embeddings(db_session)
        assert result is True
        
        # Should create/update user profile
        user_profile = db_session.query(UserProfile).first()
        assert user_profile is not None
        
        # Should have both positive and negative vectors
        pos_vector, neg_vector = get_current_profile_vectors(db_session)
        assert pos_vector is not None
        assert neg_vector is not None
        assert len(pos_vector) == 768
        assert len(neg_vector) == 768
        
        # Vectors should be normalized
        assert abs(np.linalg.norm(pos_vector) - 1.0) < 0.001
        assert abs(np.linalg.norm(neg_vector) - 1.0) < 0.001
    
    def test_profile_vectors_reflect_interaction_patterns(self, db_session, articles_with_embeddings, interactions_setup):
        """Test that profile vectors reflect the interaction patterns."""
        calculate_profile_embeddings(db_session)
        pos_vector, neg_vector = get_current_profile_vectors(db_session)
        
        # Get tech article embeddings (should be similar to positive vector)
        tech_ai_embedding = np.frombuffer(
            db_session.query(Embedding.vec).filter(
                Embedding.article_id == articles_with_embeddings['tech_ai'].article_id
            ).scalar(),
            dtype=np.float32
        )
        
        # Get finance article embeddings (should be similar to negative vector)
        finance_markets_embedding = np.frombuffer(
            db_session.query(Embedding.vec).filter(
                Embedding.article_id == articles_with_embeddings['finance_markets'].article_id
            ).scalar(),
            dtype=np.float32
        )
        
        # Positive vector should be more similar to tech articles
        tech_pos_similarity = cosine_similarity(pos_vector, tech_ai_embedding)
        finance_pos_similarity = cosine_similarity(pos_vector, finance_markets_embedding)
        assert tech_pos_similarity > finance_pos_similarity
        
        # Negative vector should be more similar to finance articles
        tech_neg_similarity = cosine_similarity(neg_vector, tech_ai_embedding)
        finance_neg_similarity = cosine_similarity(neg_vector, finance_markets_embedding)
        assert finance_neg_similarity > tech_neg_similarity


class TestArticleScoring:
    """Test article similarity scoring and recommendation logic."""
    
    def test_article_similarity_score_calculation(self):
        """Test the similarity score calculation formula."""
        # Create test vectors
        article_vec = np.array([1.0, 0.0, 0.0])
        pos_vec = np.array([1.0, 0.0, 0.0])  # Same as article
        neg_vec = np.array([0.0, 1.0, 0.0])  # Orthogonal to article
        
        score = calculate_article_similarity_score(article_vec, pos_vec, neg_vec)
        # Should be: 1.0 (positive similarity) - 0.7 * 0.0 (negative similarity) = 1.0
        assert abs(score - 1.0) < 0.001
        
        # Test with negative article
        article_vec = np.array([0.0, 1.0, 0.0])  # Same as negative vector
        score = calculate_article_similarity_score(article_vec, pos_vec, neg_vec)
        # Should be: 0.0 (positive similarity) - 0.7 * 1.0 (negative similarity) = -0.7
        expected_score = -NEGATIVE_WEIGHT_FACTOR
        assert abs(score - expected_score) < 0.001
    
    def test_calculate_article_embedding_weights(self, db_session, articles_with_embeddings, interactions_setup):
        """Test article weight calculation for all articles."""
        # First calculate profile embeddings
        calculate_profile_embeddings(db_session)
        
        # Then calculate article weights
        processed_count = calculate_article_embedding_weights(db_session)
        assert processed_count == len(DUMMY_ARTICLES)
        
        # Check that all articles have similarity scores
        articles = db_session.query(Article).all()
        for article in articles:
            assert article.vector_similarity_score is not None
            assert -1.0 <= article.vector_similarity_score <= 1.0
    
    def test_recommendation_logic_works_correctly(self, db_session, articles_with_embeddings, interactions_setup):
        """Test that the recommendation logic produces expected rankings."""
        # Calculate profile and article scores
        calculate_profile_embeddings(db_session)
        calculate_article_embedding_weights(db_session)
        
        # Get articles by category
        tech_articles = [articles_with_embeddings['tech_ai'], articles_with_embeddings['tech_quantum']]
        health_articles = [articles_with_embeddings['health_nutrition'], articles_with_embeddings['health_exercise']]
        environment_articles = [articles_with_embeddings['environment_climate'], articles_with_embeddings['environment_renewable']]
        finance_articles = [articles_with_embeddings['finance_markets'], articles_with_embeddings['finance_crypto']]
        
        # Refresh articles to get updated scores
        for article in tech_articles + health_articles + environment_articles + finance_articles:
            db_session.refresh(article)
        
        # Tech articles should have high scores (positive interactions)
        tech_scores = [article.vector_similarity_score for article in tech_articles]
        
        # Finance articles should have low scores (negative interactions)
        finance_scores = [article.vector_similarity_score for article in finance_articles]
        
        # Health and environment articles should have intermediate scores (no explicit interactions)
        health_scores = [article.vector_similarity_score for article in health_articles]
        environment_scores = [article.vector_similarity_score for article in environment_articles]
        
        # Assertions: Most importantly, tech (positive) > finance (negative)
        avg_tech_score = np.mean(tech_scores)
        avg_health_score = np.mean(health_scores)
        avg_environment_score = np.mean(environment_scores)
        avg_finance_score = np.mean(finance_scores)
        
        # Primary assertion: Articles with positive interactions should score higher than those with negative interactions
        assert avg_tech_score > avg_finance_score
        
        # Tech articles should generally score well (positive interactions)
        assert avg_tech_score > 0
        
        # Finance articles should generally score poorly (negative interactions)
        assert avg_finance_score < avg_tech_score
        
        print(f"Average scores - Tech: {avg_tech_score:.3f}, Health: {avg_health_score:.3f}, "
              f"Environment: {avg_environment_score:.3f}, Finance: {avg_finance_score:.3f}")
    
    def test_update_single_article_score(self, db_session, articles_with_embeddings, interactions_setup):
        """Test updating a single article's score."""
        # Calculate profile first
        calculate_profile_embeddings(db_session)
        
        # Update single article
        tech_article = articles_with_embeddings['tech_ai']
        result = update_single_article_score(db_session, tech_article.article_id)
        assert result is True
        
        # Check that the article has a score
        db_session.refresh(tech_article)
        assert tech_article.vector_similarity_score is not None
        assert -1.0 <= tech_article.vector_similarity_score <= 1.0


class TestConvenienceFunctions:
    """Test convenience functions for scoring articles."""
    
    def test_calculate_scores_for_recent_uninteracted_articles(self, db_session, articles_with_embeddings, interactions_setup):
        """Test scoring only recent articles without explicit interactions."""
        # Calculate profile first
        calculate_profile_embeddings(db_session)
        
        # This should score articles that don't have explicit interactions
        processed_count = calculate_scores_for_recent_uninteracted_articles(db_session)
        
        # Should process environment and health articles (no explicit interactions)
        # Should NOT process tech and finance articles (have explicit interactions)
        assert processed_count == 4  # health_nutrition, health_exercise, environment_climate, environment_renewable
        
        # Check that uninteracted articles have scores
        health_nutrition = articles_with_embeddings['health_nutrition']
        db_session.refresh(health_nutrition)
        assert health_nutrition.vector_similarity_score is not None
    
    def test_mark_interacted_articles_low_score(self, db_session, articles_with_embeddings, interactions_setup):
        """Test marking previously interacted articles with low scores."""
        marked_count = mark_interacted_articles_low_score(db_session)
        
        # Should mark articles with explicit interactions
        assert marked_count > 0
        
        # Check that interacted articles have low scores
        tech_ai = articles_with_embeddings['tech_ai']
        finance_markets = articles_with_embeddings['finance_markets']
        
        db_session.refresh(tech_ai)
        db_session.refresh(finance_markets)
        
        assert tech_ai.vector_similarity_score == -2.0
        assert finance_markets.vector_similarity_score == -2.0
        
        # Non-interacted articles should not be marked
        environment_climate = articles_with_embeddings['environment_climate']
        db_session.refresh(environment_climate)
        assert environment_climate.vector_similarity_score != -2.0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_calculate_scores_with_no_profile(self, db_session, articles_with_embeddings):
        """Test calculating article scores when no user profile exists."""
        # Don't calculate profile embeddings first
        processed_count = calculate_article_embedding_weights(db_session)
        assert processed_count == 0  # Should not process any articles
    
    def test_profile_calculation_with_old_interactions(self, db_session, sample_feed):
        """Test that very old interactions are ignored."""
        # Create an article
        old_article = Article(
            feed_id=sample_feed.feed_id,
            guid="old_guid",
            url="https://old.example.com",
            title="Old Article",
            pub_date=datetime.now(timezone.utc) - timedelta(days=100)  # Very old
        )
        db_session.add(old_article)
        db_session.commit()
        db_session.refresh(old_article)
        
        # Create old embedding
        old_embedding = create_mock_embedding('tech', 'old')
        embedding = Embedding(
            article_id=old_article.article_id,
            vec=old_embedding.tobytes()
        )
        db_session.add(embedding)
        
        # Create very old interaction (beyond LOOKBACK_DAYS)
        old_interaction = Interaction(
            article_id=old_article.article_id,
            action=ActionEnum.upvote,
            event_ts=datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS + 10)
        )
        db_session.add(old_interaction)
        db_session.commit()
        
        # Calculate profile - should ignore old interaction
        result = calculate_profile_embeddings(db_session)
        assert result is True
        
        # Should have empty vectors since old interaction is ignored
        pos_vector, neg_vector = get_current_profile_vectors(db_session)
        assert pos_vector is None
        assert neg_vector is None
    
    def test_handle_missing_embeddings(self, db_session, sample_feed):
        """Test handling articles without embeddings."""
        # Create article without embedding
        article_no_embedding = Article(
            feed_id=sample_feed.feed_id,
            guid="no_embed_guid",
            url="https://no-embed.example.com",
            title="No Embedding Article",
            pub_date=datetime.now(timezone.utc)
        )
        db_session.add(article_no_embedding)
        db_session.commit()
        
        # Try to calculate weights - should handle gracefully
        processed_count = calculate_article_embedding_weights(db_session)
        assert processed_count == 0  # No articles with embeddings to process 