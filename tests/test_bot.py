import os
import pytest
from unittest.mock import patch, MagicMock, mock_open

# Since bot.py is in the parent directory, we need to import it properly.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bot


# 1. Configuration & Environment Validation
def test_missing_env_vars_raise_exception():
    # Test missing GROQ_API_KEY using dry_run=True (skips Twitter and Telegram credentials check)
    with patch.object(bot, 'GROQ_API_KEY', None):
        with pytest.raises(ValueError, match="Missing required environment variables: GROQ_API_KEY"):
            bot.validate_environment(dry_run=True)

    # Test missing Twitter/Telegram API credentials and Gemini key without dry_run
    with patch.object(bot, 'API_KEY', None), \
         patch.object(bot, 'API_SECRET_KEY', None), \
         patch.object(bot, 'GEMINI_API_KEY', None), \
         patch.object(bot, 'TELEGRAM_BOT_TOKEN', None):
        with pytest.raises(ValueError) as excinfo:
            bot.validate_environment(dry_run=False)
        assert "API_KEY" in str(excinfo.value)
        assert "API_SECRET_KEY" in str(excinfo.value)
        assert "GEMINI_API_KEY" in str(excinfo.value)
        assert "TELEGRAM_BOT_TOKEN" in str(excinfo.value)


# 2. Payload Ingestion & Formatting
@patch("bot.requests.get")
def test_fetch_latest_leetcode_commit_success(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"sha": "12345abc", "commit": {"message": "Solve Two Sum"}}]
    mock_get.return_value = mock_response

    commit = bot.fetch_latest_leetcode_commit()
    assert commit is not None
    assert commit["sha"] == "12345abc"
    assert commit["commit"]["message"] == "Solve Two Sum"

@patch("bot.requests.get")
def test_fetch_latest_leetcode_commit_empty(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = []
    mock_get.return_value = mock_response

    commit = bot.fetch_latest_leetcode_commit()
    assert commit is None

@patch("bot.requests.get")
def test_fetch_latest_leetcode_commit_rate_limited(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_get.return_value = mock_response

    commit = bot.fetch_latest_leetcode_commit()
    assert commit is None


# 3. LLM Fallback & Resilience
@patch("bot.genai.Client")
@patch("bot.OpenAI")
def test_generate_tweet_gemini_success(mock_openai, mock_genai):
    mock_genai_client = MagicMock()
    mock_genai.return_value = mock_genai_client
    mock_genai_client.models.generate_content.return_value.text = "This is a short tech update."

    tweet = bot.generate_tweet_text("Solve Two Sum")
    assert tweet == "This is a short tech update."
    mock_openai.assert_not_called()

@patch("bot.genai.Client")
@patch("bot.OpenAI")
def test_generate_tweet_fallback_to_groq(mock_openai, mock_genai):
    # Make Gemini raise an exception
    mock_genai_client = MagicMock()
    mock_genai.return_value = mock_genai_client
    mock_genai_client.models.generate_content.side_effect = Exception("Quota exceeded")

    mock_groq_client = MagicMock()
    mock_openai.return_value = mock_groq_client
    mock_groq_client.chat.completions.create.return_value.choices[0].message.content = "Groq generated update."

    tweet = bot.generate_tweet_text("Solve Two Sum")
    assert tweet == "Groq generated update."
    mock_openai.assert_called_once()
    mock_groq_client.chat.completions.create.assert_called_once()
    kwargs = mock_groq_client.chat.completions.create.call_args[1]
    assert kwargs["model"] == "llama-3.3-70b-versatile"

@patch("bot.genai.Client")
def test_generate_tweet_trim_exceeding_length(mock_genai):
    mock_genai_client = MagicMock()
    mock_genai.return_value = mock_genai_client
    long_tweet = "A" * 300
    mock_genai_client.models.generate_content.return_value.text = long_tweet

    tweet = bot.generate_tweet_text("Solve Two Sum")
    assert len(tweet) <= 280
    assert tweet.endswith("...")
    assert tweet == ("A" * 277) + "..."


# 4. Deduplication & File Logging
@patch("bot.Path.exists")
@patch("bot.Path.rglob")
@patch("builtins.open", new_callable=mock_open, read_data="Commit ID: 12345abc\nProblem: Solve Two Sum\n\nTweet:\nCool tweet")
def test_is_already_posted(mock_file, mock_rglob, mock_exists):
    mock_exists.return_value = True
    # mock_rglob returns an iterator with one file
    mock_rglob.return_value = ["dummy_path/tweet.md"]

    assert bot.is_already_posted("12345abc") is True
    assert bot.is_already_posted("99999xyz") is False

@patch("bot.Path.mkdir")
@patch("bot.datetime")
@patch("builtins.open", new_callable=mock_open)
def test_log_and_commit_dry_run(mock_file, mock_datetime, mock_mkdir):
    # Setup mock datetime
    mock_now = MagicMock()
    mock_now.strftime.return_value = "2026-10-02_12-00-00"
    mock_datetime.now.return_value = mock_now
    
    bot.log_and_commit("12345abc", "Solve Two Sum", "Cool tweet", dry_run=True)
    
    mock_file.assert_called_once()
    handle = mock_file()
    handle.write.assert_called_once_with(
        "Commit ID: 12345abc\nProblem: Solve Two Sum\n\nTweet:\nCool tweet\n"
    )

# Notifications & Error Handling
@patch("bot.requests.post")
def test_send_telegram_message_success(mock_post):
    with patch.object(bot, 'TELEGRAM_BOT_TOKEN', 'test_token'), \
         patch.object(bot, 'TELEGRAM_CHAT_ID', 'test_id'):
        bot.send_telegram_message("Test message")
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.telegram.org/bottest_token/sendMessage"
        assert kwargs["json"] == {"chat_id": "test_id", "text": "Test message"}

@patch("bot.send_telegram_message")
@patch("bot.post_to_twitter")
@patch("bot.generate_tweet_text")
@patch("bot.fetch_latest_leetcode_commit")
@patch("bot.validate_environment")
@patch("bot.argparse.ArgumentParser.parse_args")
def test_main_fatal_error_sends_telegram(mock_args, mock_validate, mock_fetch, mock_generate, mock_post, mock_telegram):
    mock_args.return_value.dry_run = False
    mock_fetch.return_value = {"sha": "123", "commit": {"message": "Test"}}
    
    # Force an exception during tweet generation
    mock_generate.side_effect = Exception("Simulated fatal error")
    
    with pytest.raises(Exception, match="Simulated fatal error"):
        bot.main()
        
    mock_telegram.assert_called_once_with("Fatal Error in Twitter Bot: Simulated fatal error")

@patch("bot.send_telegram_message")
@patch("bot.log_and_commit")
@patch("bot.post_to_twitter")
@patch("bot.generate_tweet_text")
@patch("bot.fetch_latest_leetcode_commit")
@patch("bot.is_already_posted")
@patch("bot.validate_environment")
@patch("bot.argparse.ArgumentParser.parse_args")
def test_main_success_sends_telegram(mock_args, mock_validate, mock_is_posted, mock_fetch, mock_generate, mock_post, mock_log, mock_telegram):
    mock_args.return_value.dry_run = False
    mock_fetch.return_value = {"sha": "123", "commit": {"message": "Test Problem"}}
    mock_is_posted.return_value = False
    mock_generate.return_value = "Generated tweet"
    
    # Mock post_to_twitter to return a tweet ID
    mock_post.return_value = "987654321"
    
    bot.main()
    
    mock_telegram.assert_called_once_with("Successfully posted Test Problem: https://x.com/user/status/987654321")


# 5. CI/CD Pre-flight Check
def test_ci_cd_workflow_git_config():
    workflow_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".github", "workflows", "twitter_bot.yml")
    assert os.path.exists(workflow_path), f"Workflow file does not exist at {workflow_path}"
    with open(workflow_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert 'git config --global user.name "github-actions[bot]"' in content
        assert 'git config --global user.email "github-actions[bot]@users.noreply.github.com"' in content
