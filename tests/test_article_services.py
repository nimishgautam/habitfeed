from datetime import datetime
import hashlib
from unittest.mock import MagicMock, patch

from db.models import Article
from services.article_services import fetch_article


def test_fetch_article_success():
    mock_db_session = MagicMock()
    
    test_url = "http://test.com/article1"
    article = Article(url=test_url)
    
    mock_goose_article = MagicMock()
    mock_goose_article.title = "Test Title"
    mock_goose_article.cleaned_text = "This is the article text."
    mock_goose_article.publish_datetime_utc = datetime(2023, 1, 1, 12, 0, 0)

    with patch('services.article_services.Goose') as mock_goose, \
         patch('services.article_services.rate_limiter.wait_if_needed'):
        mock_goose_instance = mock_goose.return_value
        mock_goose_instance.extract.return_value = mock_goose_article
        
        result = fetch_article(article, mock_db_session)
        
        mock_goose_instance.extract.assert_called_once_with(url=test_url)
    
    assert result is True
    assert article.title == "Test Title"
    assert article.full_text == "This is the article text."
    assert article.pub_date == datetime(2023, 1, 1, 12, 0, 0)
    expected_hash = hashlib.sha1("This is the article text.".encode('utf-8')).hexdigest()
    assert article.content_hash == expected_hash
    
    mock_db_session.add.assert_called_once_with(article)

def test_fetch_article_no_publish_date():
    mock_db_session = MagicMock()
    
    test_url = "http://test.com/article2"
    article = Article(url=test_url)
    
    mock_goose_article = MagicMock()
    mock_goose_article.title = "Another Title"
    mock_goose_article.cleaned_text = "More article text."
    mock_goose_article.publish_datetime_utc = None

    with patch('services.article_services.Goose') as mock_goose, \
         patch('services.article_services.rate_limiter.wait_if_needed'):
        mock_goose_instance = mock_goose.return_value
        mock_goose_instance.extract.return_value = mock_goose_article
        
        result = fetch_article(article, mock_db_session)
    
    assert result is True
    assert article.title == "Another Title"
    assert article.full_text == "More article text."
    assert article.pub_date is not None # Should be set to datetime.now(timezone.utc)
    
    mock_db_session.add.assert_called_once_with(article)

def test_fetch_article_extraction_error():
    mock_db_session = MagicMock()
    
    test_url = "http://test.com/article_error"
    article = Article(url=test_url)
    
    with patch('services.article_services.Goose') as mock_goose, \
         patch('services.article_services.rate_limiter.wait_if_needed'):
        mock_goose_instance = mock_goose.return_value
        mock_goose_instance.extract.side_effect = Exception("Extraction failed")
        
        result = fetch_article(article, mock_db_session)
        
    assert result is False
    assert article.title is None
    assert article.full_text is None
    assert article.language == 'SKIP'  # Should be marked as SKIP
    
    # Should call add to mark article as SKIP and commit
    mock_db_session.add.assert_called_once_with(article)
    mock_db_session.commit.assert_called_once() 