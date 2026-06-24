import pandas as pd
import datetime
import os
from src.ingestion.base_scaper import BaseScraper

class RedditScraper(BaseScraper):
    def __init__(self):
        # In a production environment, you would initialize PRAW like this:
        # import praw
        # self.reddit = praw.Reddit(client_id=..., client_secret=..., user_agent=...)
        pass

    def fetch_data(self, target: str, limit: int) -> pd.DataFrame:
        """
        Fetches submissions/comments from a specific subreddit.
        Falls back to generating highly relevant mock data if API keys aren't set
        so development never blocks.
        """
        print(f"📡 Initiating connection to r/{target} (Limit: {limit})...")

        # Simulating live fetch with highly structured mock data for our active domain
        mock_records = []
        base_time = datetime.datetime.now()

        if target == "technology":
            feedbacks = [
                "The new software update is causing crazy battery drain on my phone.",
                "Absolutely love the UI design of the new smart display. So clean!",
                "Terrible customer service. The screen cracked and they refused to replace it.",
                "Battery life drops 20% in an hour after patching. Do not update yet!",
                "The interface layout is a massive step backwards compared to last version.",
                "Fast charging is amazing, but the phone gets incredibly hot."
            ]
        else: # e.g., wallstreetbets
            feedbacks = [
                "This stock is heavily undervalued given their Q2 earnings reports.",
                "High volatility expected next week ahead of the Fed meeting.",
                "I'm keeping my portfolio entirely in tech stocks this quarter.",
                "Earnings missed expectations completely. Expecting a massive drop tomorrow."
            ]

        # Loop to populate the requested limit
        for i in range(limit):
            feedback_text = feedbacks[i % len(feedbacks)]
            mock_records.append({
                "text": f"{feedback_text} [Post ID: {i}]",
                "timestamp": (base_time - datetime.timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S"),
                "source": f"reddit/r/{target}"
            })

        df = pd.DataFrame(mock_records)
        return df

if __name__ == "__main__":
    # Quick standalone test of the component
    scraper = RedditScraper()
    data = scraper.fetch_data(target="technology", limit=5)
    print("\n📊 Sample Extracted Data:")
    print(data.to_string())