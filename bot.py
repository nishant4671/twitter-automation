import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

import requests
import tweepy
from dotenv import load_dotenv
from git import Repo
from google import genai
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "your_github_username")
LEETCODE_REPO_NAME = os.getenv("LEETCODE_REPO_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def fetch_latest_leetcode_commit():
    """Fetch the most recent commit exclusively from the configured LeetCode repository."""
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{LEETCODE_REPO_NAME}/commits"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"GitHub API Error: {response.status_code} - {response.text}")
    
    commits = response.json()
    if not commits:
        raise ValueError("No commits found in the specified repository.")
        
    return commits[0]


def is_already_posted(commit_sha):
    """Check if the commit ID exists in any tweet.md file inside the posts/ directory."""
    posts_dir = Path("posts")
    if not posts_dir.exists():
        return False
    
    # Traverse through all subdirectories in posts/ looking for tweet.md
    for md_file in posts_dir.rglob("tweet.md"):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
                if commit_sha in content:
                    return True
        except Exception as e:
            logger.error(f"Error reading file {md_file}: {e}")
            
    return False


def generate_tweet_text(commit_msg):
    """
    Generate a tweet update under 280 characters with no emojis using Gemini,
    with an automated fallback to Grok.
    """
    prompt = (
        f"Take the following raw LeetCode commit message: '{commit_msg}'. "
        f"Synthesize a short, engaging tech update under 280 characters "
        f"without using emojis. Return only the tweet text."
    )
    
    # 1. Primary Attempt: Gemini API
    try:
        logger.info("Attempting to generate tweet using Gemini API...")
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        tweet = response.text.strip()
        
        if len(tweet) > 280:
            tweet = tweet[:277] + "..."
        return tweet
        
    except Exception as e:
        logger.warning(f"Gemini generation failed: {e}. Falling back to Groq...")
    
    # 2. Fallback Attempt: Groq API
    try:
        logger.info("Attempting to generate tweet using Groq API...")
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful coding assistant that writes short tech updates without emojis."},
                {"role": "user", "content": prompt}
            ]
        )
        tweet = response.choices[0].message.content.strip()
        
        if len(tweet) > 280:
            tweet = tweet[:277] + "..."
        return tweet
        
    except Exception as e:
        logger.error(f"Groq fallback generation also failed: {e}")
        
    # Ultimate static fallback
    return f"Solved a new problem on LeetCode: {commit_msg}"[:280]


def post_to_twitter(text):
    """Post text to Twitter API v2 Free Tier."""
    client = tweepy.Client(
        consumer_key=API_KEY,
        consumer_secret=API_SECRET_KEY,
        access_token=ACCESS_TOKEN,
        access_token_secret=ACCESS_TOKEN_SECRET
    )
    try:
        response = client.create_tweet(text=text)
        logger.info("Successfully posted tweet to Twitter.")
        return response.data['id']
    except tweepy.errors.TooManyRequests as e:
        logger.error(f"Rate limit exceeded (HTTP 429): {e}")
        raise
    except tweepy.errors.Forbidden as e:
        logger.error(f"Forbidden (HTTP 403): {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to post tweet: {e}")
        raise


def log_and_commit(commit_sha, commit_msg, tweet_text, dry_run=False):
    """
    Creates a new folder with a markdown log of the tweet and uses GitPython 
    to automatically commit and push the changes back to the origin repository.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    post_dir = Path("posts") / timestamp
    post_dir.mkdir(parents=True, exist_ok=True)
    
    md_path = post_dir / "tweet.md"
    content = f"Commit ID: {commit_sha}\nProblem: {commit_msg}\n\nTweet:\n{tweet_text}\n"
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    logger.info(f"Created Git-based log file at {md_path}")
    
    if dry_run:
        logger.info("[DRY RUN] Skipping Git commit and push operations.")
        return
        
    try:
        repo = Repo(Path.cwd())
        
        # Add and Commit
        repo.index.add([str(md_path)])
        repo.index.commit(f"docs: log automated tweet for commit {commit_sha[:7]}")
        logger.info("Created local Git commit for log entry.")
        
        # Push to remote (requires proper Git config/authentication)
        origin = repo.remote(name='origin')
        origin.push()
        logger.info("Successfully pushed Git log back to remote repository.")
    except Exception as e:
        logger.error(f"Git operations failed: {e}")


def send_telegram_message(text):
    """Send a notification message via Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials missing, skipping notification.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        logger.info("Telegram notification sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}")


def validate_environment(dry_run=False):
    """Validate that required environment variables are present."""
    missing = []
    if not dry_run:
        required_twitter = [
            'API_KEY', 'API_SECRET_KEY', 'ACCESS_TOKEN', 'ACCESS_TOKEN_SECRET',
            'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID'
        ]
        for var in required_twitter:
            if not globals().get(var):
                missing.append(var)
                
    if not GEMINI_API_KEY:
        missing.append('GEMINI_API_KEY')
    if not GROQ_API_KEY:
        missing.append('GROQ_API_KEY')
    if not LEETCODE_REPO_NAME:
        missing.append('LEETCODE_REPO_NAME')
        
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


def main():
    parser = argparse.ArgumentParser(description="LeetCode Twitter Automation Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and generate without posting or pushing changes")
    args = parser.parse_args()

    try:
        # Configuration & Environment Validation
        validate_environment(args.dry_run)
                
        # 1. Data Ingestion
        logger.info("Fetching latest LeetCode commit...")
        commit = fetch_latest_leetcode_commit()
            
        commit_sha = commit['sha']
        commit_msg = commit['commit']['message'].split('\n')[0]
        
        # 2. Duplicate Validation
        if is_already_posted(commit_sha):
            logger.info(f"Commit {commit_sha[:7]} is already present in the audit log. Exiting to prevent duplicate.")
            return
            
        logger.info(f"Found new commit: {commit_msg}")
        
        # 3. Dynamic Tweet Generation (LLM Fallback System)
        tweet_text = generate_tweet_text(commit_msg)
        
        # 4. Local Testing Mode
        if args.dry_run:
            logger.info("\n--- [DRY RUN RESULTS] ---\n")
            logger.info(f"Generated Tweet:\n{tweet_text}\n")
            log_and_commit(commit_sha, commit_msg, tweet_text, dry_run=True)
            return
            
        # 5. Production Execution
        tweet_id = post_to_twitter(tweet_text)
        log_and_commit(commit_sha, commit_msg, tweet_text, dry_run=False)
        send_telegram_message(f"Successfully posted {commit_msg}: https://x.com/user/status/{tweet_id}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        send_telegram_message(f"Fatal Error in Twitter Bot: {str(e)}")
        raise


if __name__ == "__main__":
    main()
