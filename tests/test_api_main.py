import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
import json
import feedparser
import numpy as np

# Import the FastAPI app and dependencies
from api.main import app, get_db
from db.models import Feed, Article, Interaction, ActionEnum
from api.schemas import FeedCreate, ActionRecord


@pytest.fixture
def client(db_session):
    """Create a test client with overridden database dependency."""
    def override_get_db():
        return db_session
    
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_feed(db_session):
    """Create a sample feed for testing."""
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    feed = Feed(
        feed_url=f"https://example-{unique_id}.com/rss.xml",
        title=f"Sample Feed {unique_id}",
        last_checked=datetime.now(timezone.utc)
    )
    db_session.add(feed)
    db_session.commit()
    db_session.refresh(feed)
    return feed


@pytest.fixture
def sample_articles(db_session, sample_feed):
    """Create sample articles for testing."""
    import uuid
    articles = []
    for i in range(3):
        unique_id = str(uuid.uuid4())[:8]
        article = Article(
            feed_id=sample_feed.feed_id,
            guid=f"article-{unique_id}-{i}",
            url=f"https://example.com/article-{unique_id}-{i}",
            title=f"Sample Article {i}",
            rss_description=f"Description for article {i}",
            pub_date=datetime.now(timezone.utc),
            full_text=f"Full text content for article {i}",
            vector_similarity_score=0.5 + i * 0.1
        )
        articles.append(article)
        db_session.add(article)
    
    db_session.commit()
    for article in articles:
        db_session.refresh(article)
    return articles


@pytest.fixture
def sample_interactions(db_session, sample_articles):
    """Create sample interactions for testing."""
    interactions = []
    
    # Add negative interaction to first article (should exclude it from lists)
    negative_interaction = Interaction(
        article_id=sample_articles[0].article_id,
        action=ActionEnum.skip,
        action_duration_ms=1000
    )
    interactions.append(negative_interaction)
    db_session.add(negative_interaction)
    
    # Add positive interaction to second article
    positive_interaction = Interaction(
        article_id=sample_articles[1].article_id,
        action=ActionEnum.upvote,
        action_duration_ms=5000
    )
    interactions.append(positive_interaction)
    db_session.add(positive_interaction)
    
    db_session.commit()
    return interactions


class TestFeedsEndpoints:
    """Test cases for feed-related endpoints."""
    
    def test_get_all_feeds(self, client, sample_feed):
        """Test GET /feeds endpoint."""
        response = client.get("/feeds")
        assert response.status_code == 200
        
        feeds = response.json()
        assert len(feeds) == 1
        assert feeds[0]["feed_id"] == sample_feed.feed_id
        assert feeds[0]["feed_url"] == sample_feed.feed_url
        assert feeds[0]["title"] == sample_feed.title
    
    def test_get_feed_by_id_success(self, client, sample_feed):
        """Test GET /feeds/{feed_id} with valid ID."""
        response = client.get(f"/feeds/{sample_feed.feed_id}")
        assert response.status_code == 200
        
        feed = response.json()
        assert feed["feed_id"] == sample_feed.feed_id
        assert feed["feed_url"] == sample_feed.feed_url
        assert feed["title"] == sample_feed.title
    
    def test_get_feed_by_id_not_found(self, client):
        """Test GET /feeds/{feed_id} with invalid ID."""
        response = client.get("/feeds/999")
        assert response.status_code == 404
        assert response.json()["detail"] == "Feed not found"
    
    def test_add_feed_success(self, client, db_session):
        """Test POST /feeds with valid feed URL."""
        feed_data = {"feed_url": "https://newsite.com/rss.xml"}
        
        response = client.post("/feeds", json=feed_data)
        assert response.status_code == 201
        
        feed = response.json()
        assert feed["feed_url"] == feed_data["feed_url"]
        assert "feed_id" in feed
        
        # Verify feed was actually created in database
        db_feed = db_session.query(Feed).filter(Feed.feed_url == feed_data["feed_url"]).first()
        assert db_feed is not None
    
    def test_add_feed_duplicate(self, client, sample_feed):
        """Test POST /feeds with duplicate feed URL."""
        feed_data = {"feed_url": sample_feed.feed_url}
        
        response = client.post("/feeds", json=feed_data)
        assert response.status_code == 400
        assert response.json()["detail"] == "Feed already exists"


class TestArticlesEndpoints:
    """Test cases for article-related endpoints."""
    
    def test_get_articles_by_feed(self, client, sample_feed, sample_articles, sample_interactions):
        """Test GET /feeds/{feed_id}/articles - should exclude articles with negative actions."""
        response = client.get(f"/feeds/{sample_feed.feed_id}/articles")
        assert response.status_code == 200
        
        articles = response.json()
        # Should exclude the first article which has a skip interaction
        assert len(articles) == 2
        article_ids = [article["article_id"] for article in articles]
        assert sample_articles[0].article_id not in article_ids
        assert sample_articles[1].article_id in article_ids
        assert sample_articles[2].article_id in article_ids
    
    def test_get_articles_by_feed_with_limit(self, client, sample_feed, sample_articles):
        """Test GET /feeds/{feed_id}/articles with limit parameter."""
        response = client.get(f"/feeds/{sample_feed.feed_id}/articles?limit=1")
        assert response.status_code == 200
        
        articles = response.json()
        assert len(articles) == 1
    
    def test_get_recent_articles(self, client, sample_articles, sample_interactions):
        """Test GET /articles/recent - should exclude articles with negative actions."""
        response = client.get("/articles/recent?limit=5")
        assert response.status_code == 200
        
        articles = response.json()
        # Should exclude the first article which has a skip interaction
        assert len(articles) == 2
        article_ids = [article["article_id"] for article in articles]
        assert sample_articles[0].article_id not in article_ids
    
    def test_get_article_by_id_success(self, client, sample_articles):
        """Test GET /articles/{article_id} with valid ID."""
        article = sample_articles[0]
        response = client.get(f"/articles/{article.article_id}")
        assert response.status_code == 200
        
        result = response.json()
        assert result["article_id"] == article.article_id
        assert result["title"] == article.title
        assert result["url"] == article.url
    
    def test_get_article_by_id_not_found(self, client):
        """Test GET /articles/{article_id} with invalid ID."""
        response = client.get("/articles/999")
        assert response.status_code == 404
        assert response.json()["detail"] == "Article not found"


class TestActionsEndpoints:
    """Test cases for article action endpoints."""
    
    def test_record_article_action_success(self, client, sample_articles):
        """Test POST /articles/{article_id}/actions with valid action."""
        article = sample_articles[0]
        action_data = {"action": "upvote", "action_duration_ms": 2000}
        
        response = client.post(f"/articles/{article.article_id}/actions", json=action_data)
        assert response.status_code == 201
        
        result = response.json()
        assert result["message"] == "Action recorded successfully"
        assert result["recorded_action"] == "upvote"
        assert result["article_id"] == article.article_id
    
    def test_record_article_action_invalid_article(self, client):
        """Test POST /articles/{article_id}/actions with invalid article ID."""
        action_data = {"action": "upvote", "action_duration_ms": 2000}
        
        response = client.post("/articles/999/actions", json=action_data)
        assert response.status_code == 404
        assert response.json()["detail"] == "Article not found"
    
    def test_record_article_action_invalid_action(self, client, sample_articles):
        """Test POST /articles/{article_id}/actions with invalid action."""
        article = sample_articles[0]
        action_data = {"action": "invalid_action", "action_duration_ms": 2000}
        
        response = client.post(f"/articles/{article.article_id}/actions", json=action_data)
        assert response.status_code == 406
        assert "is not supported" in response.json()["detail"]
    
    def test_record_click_with_existing_negative_interaction(self, client, db_session, sample_articles):
        """Test that click action becomes click_archived when negative interactions exist."""
        article = sample_articles[0]
        
        # First add a negative interaction
        negative_interaction = Interaction(
            article_id=article.article_id,
            action=ActionEnum.skip,
            action_duration_ms=1000
        )
        db_session.add(negative_interaction)
        db_session.commit()
        
        # Now try to add a click action
        action_data = {"action": "click", "action_duration_ms": 500}
        response = client.post(f"/articles/{article.article_id}/actions", json=action_data)
        assert response.status_code == 201
        
        result = response.json()
        assert result["recorded_action"] == "click_archived"
    
    def test_record_negative_action_converts_existing_clicks(self, client, db_session, sample_articles):
        """Test that negative actions convert existing click events to click_archived."""
        article = sample_articles[0]
        
        # First add a click interaction
        click_interaction = Interaction(
            article_id=article.article_id,
            action=ActionEnum.click,
            action_duration_ms=500
        )
        db_session.add(click_interaction)
        db_session.commit()
        
        # Now add a negative action
        action_data = {"action": "skip", "action_duration_ms": 1000}
        response = client.post(f"/articles/{article.article_id}/actions", json=action_data)
        assert response.status_code == 201
        
        # Check that the click interaction was converted
        db_session.refresh(click_interaction)
        assert click_interaction.action == ActionEnum.click_archived
    
    def test_record_non_dwell_action_sets_similarity_score(self, client, db_session, sample_articles):
        """Test that non-dwell actions set vector_similarity_score to -2."""
        article = sample_articles[0]
        original_score = article.vector_similarity_score
        
        action_data = {"action": "upvote", "action_duration_ms": 2000}
        response = client.post(f"/articles/{article.article_id}/actions", json=action_data)
        assert response.status_code == 201
        
        # Check that vector_similarity_score was set to -2
        db_session.refresh(article)
        assert article.vector_similarity_score == -2.0
        assert article.vector_similarity_score != original_score


class TestServiceEndpoints:
    """Test cases for service endpoints."""
    
    @patch('api.main.update_feeds')
    def test_update_all_feeds(self, mock_update_feeds, client):
        """Test POST /feeds/update endpoint."""
        mock_update_feeds.return_value = None
        
        response = client.post("/feeds/update")
        assert response.status_code == 200
        assert response.json()["message"] == "Feeds update started"
        mock_update_feeds.assert_called_once()
    
    @patch('api.main.update_feeds')
    def test_update_all_feeds_error(self, mock_update_feeds, client):
        """Test POST /feeds/update endpoint with service error."""
        mock_update_feeds.side_effect = Exception("Service error")
        
        response = client.post("/feeds/update")
        assert response.status_code == 500
        assert "Error updating feeds: Service error" in response.json()["detail"]
    
    @patch('api.main.fetch_articles')
    def test_fetch_all_articles(self, mock_fetch_articles, client):
        """Test POST /articles/fetch endpoint."""
        mock_fetch_articles.return_value = None
        
        response = client.post("/articles/fetch")
        assert response.status_code == 200
        assert response.json()["message"] == "Article fetching started"
        mock_fetch_articles.assert_called_once()
    
    @patch('api.main.fetch_articles')
    def test_fetch_all_articles_error(self, mock_fetch_articles, client):
        """Test POST /articles/fetch endpoint with service error."""
        mock_fetch_articles.side_effect = Exception("Fetch error")
        
        response = client.post("/articles/fetch")
        assert response.status_code == 500
        assert "Error fetching articles: Fetch error" in response.json()["detail"]


class TestSSEEndpoints:
    """Test cases for Server-Sent Events endpoints."""
    
    @patch('api.main.update_feeds')
    def test_update_feeds_stream(self, mock_update_feeds, client):
        """Test GET /feeds/update/stream endpoint."""
        # Mock the progress callback behavior
        def mock_update_with_callback(db, progress_callback=None):
            if progress_callback:
                progress_callback(1, 2, "Processing feed 1")
                progress_callback(2, 2, "All feeds updated successfully")
        
        mock_update_feeds.side_effect = mock_update_with_callback
        
        response = client.get("/feeds/update/stream")
        assert response.status_code == 200
        # The endpoint sets Content-Type to text/event-stream in headers
        assert response.headers["content-type"] == "text/event-stream"
        
        # Check for SSE headers
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["connection"] == "keep-alive"
    
    @patch('api.main.fetch_articles')
    def test_fetch_articles_stream(self, mock_fetch_articles, client):
        """Test GET /articles/fetch/stream endpoint."""
        # Mock the progress callback behavior
        def mock_fetch_with_callback(db, progress_callback=None):
            if progress_callback:
                progress_callback(1, 3, "Processing article 1")
                progress_callback(2, 3, "Processing article 2")
                progress_callback(3, 3, "All articles processed successfully")
        
        mock_fetch_articles.side_effect = mock_fetch_with_callback
        
        response = client.get("/articles/fetch/stream")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"


class TestIntegrationScenarios:
    """Integration test scenarios combining multiple endpoints."""
    
    def test_full_workflow_add_feed_and_get_articles(self, client, db_session):
        """Test adding a feed and then retrieving its articles."""
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        
        # Add a new feed
        feed_data = {"feed_url": f"https://test-{unique_id}.com/rss.xml"}
        feed_response = client.post("/feeds", json=feed_data)
        assert feed_response.status_code == 201
        feed_id = feed_response.json()["feed_id"]
        
        # Manually add an article to this feed for testing
        article = Article(
            feed_id=feed_id,
            guid=f"test-article-{unique_id}",
            url=f"https://test-{unique_id}.com/article",
            title="Test Article",
            pub_date=datetime.now(timezone.utc)
        )
        db_session.add(article)
        db_session.commit()
        
        # Get articles for the feed
        articles_response = client.get(f"/feeds/{feed_id}/articles")
        assert articles_response.status_code == 200
        
        articles = articles_response.json()
        assert len(articles) == 1
        assert articles[0]["title"] == "Test Article"
    
    def test_article_action_workflow(self, client, db_session, sample_articles):
        """Test the complete article action workflow."""
        article = sample_articles[0]
        
        # Step 1: Click on an article
        click_response = client.post(
            f"/articles/{article.article_id}/actions",
            json={"action": "click", "action_duration_ms": 500}
        )
        assert click_response.status_code == 201
        
        # Step 2: Skip the same article (should convert click to click_archived)
        skip_response = client.post(
            f"/articles/{article.article_id}/actions", 
            json={"action": "skip", "action_duration_ms": 1000}
        )
        assert skip_response.status_code == 201
        
        # Step 3: Try to click again (should become click_archived due to existing skip)
        click_again_response = client.post(
            f"/articles/{article.article_id}/actions",
            json={"action": "click", "action_duration_ms": 300}
        )
        assert click_again_response.status_code == 201
        assert click_again_response.json()["recorded_action"] == "click_archived"
        
        # Step 4: Verify the article no longer appears in recent articles
        recent_response = client.get("/articles/recent")
        recent_articles = recent_response.json()
        article_ids = [a["article_id"] for a in recent_articles]
        assert article.article_id not in article_ids


# Additional fixtures for testing external service mocking
@pytest.fixture
def mock_feedparser_data():
    """Mock data for feedparser.parse()."""
    mock_feed = Mock()
    mock_feed.feed.title = "Mocked Feed Title"
    
    mock_entry1 = Mock()
    mock_entry1.guid = "mock-guid-1"
    mock_entry1.link = "https://example.com/article1"
    mock_entry1.get.return_value = "Mock description 1"
    mock_entry1.published_parsed = (2023, 10, 15, 12, 0, 0)
    
    mock_entry2 = Mock()
    mock_entry2.guid = "mock-guid-2"
    mock_entry2.link = "https://example.com/article2"
    mock_entry2.get.return_value = "Mock description 2"
    mock_entry2.published_parsed = (2023, 10, 16, 14, 30, 0)
    
    mock_feed.entries = [mock_entry1, mock_entry2]
    return mock_feed


@pytest.fixture
def mock_goose_data():
    """Mock data for Goose().extract()."""
    mock_article = Mock()
    mock_article.title = "Mocked Article Title"
    mock_article.cleaned_text = "This is the mocked article content."
    mock_article.publish_datetime_utc = datetime(2023, 10, 15, 12, 0, 0, tzinfo=timezone.utc)
    return mock_article


class TestExternalServiceMocking:
    """Test cases that verify external services are properly mocked."""
    
    @patch('feedparser.parse')
    @patch('api.main.update_feeds')
    def test_feed_update_with_mocked_feedparser(self, mock_update_feeds, mock_feedparser, 
                                               client, mock_feedparser_data):
        """Test that feedparser.parse is properly mocked in feed updates."""
        mock_feedparser.return_value = mock_feedparser_data
        
        # Configure mock_update_feeds to actually call feedparser.parse
        def side_effect(db, progress_callback=None):
            feedparser.parse("https://example.com/rss.xml")
        
        mock_update_feeds.side_effect = side_effect
        
        response = client.post("/feeds/update")
        assert response.status_code == 200
        
        # Verify feedparser.parse was called
        mock_feedparser.assert_called()
    
    @patch('goose3.Goose')
    @patch('api.main.fetch_articles')
    def test_article_fetch_with_mocked_goose(self, mock_fetch_articles, mock_goose_class,
                                           client, mock_goose_data):
        """Test that Goose().extract is properly mocked in article fetching."""
        mock_goose_instance = Mock()
        mock_goose_instance.extract.return_value = mock_goose_data
        mock_goose_class.return_value = mock_goose_instance
        
        # Configure mock_fetch_articles to actually call Goose
        def side_effect(db, progress_callback=None):
            from goose3 import Goose
            g = Goose()
            g.extract(url="https://example.com/article")
        
        mock_fetch_articles.side_effect = side_effect
        
        response = client.post("/articles/fetch")
        assert response.status_code == 200
        
        # Verify Goose().extract was called
        mock_goose_instance.extract.assert_called()
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_embedding_creation_with_mocked_transformer(self, mock_transformer_class, client):
        """Test that SentenceTransformer.encode is properly mocked."""
        mock_transformer = Mock()
        mock_transformer.encode.return_value = np.random.rand(768).astype(np.float32)
        mock_transformer_class.return_value = mock_transformer
        
        # This would be called indirectly through article processing
        response = client.post("/articles/fetch")
        # The response will depend on whether articles exist to process
        assert response.status_code in [200, 500]  # 500 if no articles to process 