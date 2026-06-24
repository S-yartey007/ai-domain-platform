from abc import ABC, abstractmethod
import pandas as pd


class BaseScraper(ABC):
    """
    Abstract Base Class that enforces a consistent interface
    for all data ingestion modules in the platform.
    """

    @abstractmethod
    def fetch_data(self, target: str, limit: int) -> pd.DataFrame:
        """
        Fetches raw data from a source and converts it into a uniform pandas DataFrame.

        Parameters:
        - target (str): The specific target identifier (e.g., subreddit name, app ID).
        - limit (int): The maximum number of items to fetch.

        Returns:
        - pd.DataFrame: A DataFrame containing at minimum columns: ['text', 'timestamp', 'source']
        """
        pass