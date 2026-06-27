import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from openai import OpenAI

class TopicTrackerEngine:
    def __init__(self, llm_prompt: str, min_clusters: int = 3):
        self.llm_prompt = llm_prompt
        self.min_clusters = min_clusters
        # Initialize OpenAI client (looks for OPENAI_API_KEY environment variable)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "mock_key"))

    def cluster_and_label(self, csv_path: str) -> pd.DataFrame:
        """Loads text data, groups them into semantic clusters, and returns labeled topics."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No data file found at {csv_path}. Run ingestion first!")

        df = pd.read_csv(csv_path)
        if len(df) < self.min_clusters:
            raise ValueError("Not enough data rows to form clusters.")

        print(f"🔮 Vectorizing and clustering {len(df)} text entries...")

        # 1. Classical ML: Vectorize text using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        tfidf_matrix = vectorizer.fit_transform(df['text'].astype(str))

        # 2. Unsupervised Clustering: K-Means
        kmeans = KMeans(n_clusters=self.min_clusters, random_state=42, n_init='auto')
        df['cluster_id'] = kmeans.fit_predict(tfidf_matrix)

        # 3. GenAI Layer: Label the clusters using an LLM
        cluster_labels = {}
        print("🤖 Generating AI titles for discovered text clusters...")

        for cluster_num in range(self.min_clusters):
            # Extract top text examples belonging to this cluster
            cluster_samples = df[df['cluster_id'] == cluster_num]['text'].head(3).tolist()
            samples_text = "\n- ".join(cluster_samples)

            # Check if valid API Key exists, otherwise fall back to elegant mock labels
            if os.environ.get("OPENAI_API_KEY"):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": self.llm_prompt},
                            {"role": "user", "content": f"Sample texts from this cluster:\n- {samples_text}"}
                        ],
                        max_tokens=15,
                        temperature=0.3
                    )
                    if response.choices[0].message.content:
                        topic_title = response.choices[0].message.content.strip()
                    else:
                        topic_title = ""

                except Exception as e:
                    topic_title = f"Cluster {cluster_num} (API Error)"
            else:
                # Intelligent mock naming fallback based on keywords found in the text sample
                sample_str = samples_text.lower()
                if "battery" in sample_str or "drain" in sample_str:
                    topic_title = "Battery Drain Issues"
                elif "ui" in sample_str or "interface" in sample_str or "display" in sample_str:
                    topic_title = "UI/UX Layout Praise"
                elif "earnings" in sample_str or "undervalued" in sample_str:
                    topic_title = "Stock Market Volatility"
                else:
                    topic_title = f"General Cluster {cluster_num} Feedback"

            cluster_labels[cluster_num] = topic_title
            print(f"  📍 Cluster {cluster_num} mapped to title: '{topic_title}'")

        # Map cluster IDs to their newly generated human-friendly titles
        df['assigned_topic'] = df['cluster_id'].map(cluster_labels)
        return df

if __name__ == "__main__":
    # Test stub to quickly verify engine functionality standalone
    print("🧪 Testing Topic Tracker Engine Standalone...")
    mock_prompt = "Summarize these comments into a 3-word title."
    tracker = TopicTrackerEngine(llm_prompt=mock_prompt, min_clusters=3)

    # Try running on the ingested tech file if it exists
    test_file = "data/raw_scraped/tech_gadgets_raw.csv"
    if os.path.exists(test_file):
        processed_df = tracker.cluster_and_label(test_file)
        print("\n📊 Sample Clustered Output Preview:")
        print(processed_df[['text', 'cluster_id', 'assigned_topic']].head(5))
    else:
        print(f"❌ Could not run test; {test_file} does not exist yet. Run ingestion first.")