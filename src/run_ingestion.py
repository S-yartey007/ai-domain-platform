import os
import pandas as pd
from src.utils.config_loader import load_config
from src.ingestion.reddit_worker import RedditScraper

def main():
    print("🚀 Starting Data Ingestion Pipeline...")

    # 1. Load active configuration
    try:
        config = load_config()
        domain_key = config["domain_key"]
        print(f"📦 Active Domain Config: {config['name']} ({domain_key})")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return

    # 2. Extract scraper configuration settings
    scraper_cfg = config.get("scraper", {})
    source_type = scraper_cfg.get("source_type")
    target = scraper_cfg.get("target_subreddit")
    limit = scraper_cfg.get("limit", 100)

    # 3. Factory Pattern: Instantiate the matching scraper
    if source_type == "reddit":
        scraper = RedditScraper()
    else:
        print(f"❌ Unknown scraper source type: '{source_type}' configured.")
        return

    # 4. Fetch the data
    try:
        df = scraper.fetch_data(target=target, limit=limit)
        print(f"✅ Successfully fetched {len(df)} records.")
    except Exception as e:
        print(f"❌ Error during data fetching: {e}")
        return

    # 5. Ensure the storage directory exists and save the data
    output_dir = "data/raw_scraped"
    os.makedirs(output_dir, exist_ok=True)

    # Save using the domain name so datasets stay separated cleanly!
    output_path = os.path.join(output_dir, f"{domain_key}_raw.csv")
    df.to_csv(output_path, index=False)
    print(f"💾 Raw data saved securely to: {output_path}\n")

if __name__ == "__main__":
    main()