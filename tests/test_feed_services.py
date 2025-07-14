from services.feed_services import update_feed
from db.models import Feed, Article
from sqlalchemy.orm import Session
from unittest.mock import MagicMock, patch

class FeedParserEntry:
    """A helper class to mock feedparser entries."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

@patch('services.feed_services.feedparser.parse')
def test_update_feed_adds_articles_without_title(mock_feedparser, db_session: Session):
    """
    Tests that update_feed adds new articles from a feed but leaves the title blank.
    """
    # 1. Setup a mock for feedparser
    mock_parsed_feed = MagicMock()
    mock_parsed_feed.feed.title = "Test Feed"
    mock_parsed_feed.entries = [
        FeedParserEntry(
            guid="guid1",
            link="http://test.com/1",
            title="Article 1",
            description="Article 1 description",
            published_parsed=(2023, 1, 1, 12, 0, 0, 0, 1, 0)
        ),
        FeedParserEntry(
            guid="guid2",
            link="http://test.com/2",
            title="Article 2",
            description="Article 2 description",
            published_parsed=(2023, 1, 2, 12, 0, 0, 0, 2, 0)
        ),
    ]
    mock_feedparser.return_value = mock_parsed_feed

    # 2. Setup initial state in the DB
    feed_url = "http://test.com/rss.xml"
    feed = Feed(feed_url=feed_url)
    db_session.add(feed)
    db_session.commit()
    db_session.refresh(feed)

    initial_article_count = db_session.query(Article).count()
    initial_titled_article_count = db_session.query(Article).filter(Article.title != None).count()

    # 3. Run the function to be tested
    update_feed(feed, db_session)

    # 4. Assertions
    # Check that article count increased by 2
    final_article_count = db_session.query(Article).count()
    assert final_article_count == initial_article_count + 2

    # Check that no new articles have titles
    final_titled_article_count = db_session.query(Article).filter(Article.title != None).count()
    assert final_titled_article_count == initial_titled_article_count

    # Verify that the feed title and last_checked were updated
    updated_feed = db_session.query(Feed).filter(Feed.feed_id == feed.feed_id).one()
    assert updated_feed.title == "Test Feed"
    assert updated_feed.last_checked is not None 