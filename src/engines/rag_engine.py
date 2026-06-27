import os
import chromadb
import pandas as pd
from openai import OpenAI

class RAGEngine:
    def __init__(self, collection_name: str, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 1. Initialize local persistent database storage
        self.db_client = chromadb.PersistentClient(path="data/vector_store")

        # Create or fetch the specific domain data collection
        self.collection = self.db_client.get_or_create_collection(name=collection_name)
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "mock_key"))

    def ingest_dataframe(self, df: pd.DataFrame, text_column: str = "text"):
        """Breaks a dataframe's text into chunks and upserts them into ChromaDB."""
        print(f"📦 Preparing to ingest data into Vector Store collection...")

        documents = []
        metadatas = []
        ids = []

        for idx, row in df.iterrows():
            text = str(row[text_column])

            # Simple character-based chunking mechanism
            start = 0
            chunk_idx = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk = text[start:end]

                documents.append(chunk)
                metadatas.append({"original_row": idx, "source": row.get("source", "unknown")})
                ids.append(f"doc_{idx}_chunk_{chunk_idx}")

                start += (self.chunk_size - self.chunk_overlap)
                chunk_idx += 1

        if documents:
            # Upsert into ChromaDB
            # (Note: ChromaDB uses a built-in lightweight embedding model by default if none is provided!)
            self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
            print(f"✅ Successfully vectorized and stored {len(documents)} chunks in database.")

    def query(self, user_question: str, num_results: int = 3) -> str:
        """Retrieves matching chunks and builds an LLM-synthesized response."""
        print(f"🔍 Searching vector database for: '{user_question}'")

        # 2. Query the vector database
        results = self.collection.query(query_texts=[user_question], n_results=num_results)

        # Flatten retrieved document chunks into a single context string
        if results is None:
            results = {}
        # 2. Extract "documents". If it's missing OR if it's explicitly None, fallback to [[]]
        documents = results.get("documents")
        if documents is None:
            documents = [[]]

            # 3. Safe to grab the first element now!
        retrieved_chunks = documents[0]


        context = "\n---\n".join(retrieved_chunks)

        if not retrieved_chunks:
            return "I couldn't find any relevant data points in my database to answer that question."

        # 3. LLM Prompt Synthesis
        system_prompt = (
            "You are an expert intelligence analyst. Answer the user's question using ONLY the provided "
            "document context below. If the answer cannot be found in the context, say 'I cannot confirm "
            "this with current context data.' Do not hallucinate."
        )
        user_content = f"Context Data:\n{context}\n\nQuestion: {user_question}"

        # API check or intelligent mock response builder
        if os.environ.get("OPENAI_API_KEY"):
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.0
                )
                if response.choices[0].message.content is not None:
                   return response.choices[0].message.content.strip()
                else:
                    return ""
            except Exception as e:
                return f"Error communicating with LLM engine: {e}"
        else:
            # Context-Aware Mock Generator for testing offline
            context_lower = context.lower()
            if "battery" in user_question.lower() and "drain" in context_lower:
                return "Based on database tracking, users report severe battery drain issues occurring right after patching the latest software update."
            elif "ui" in user_question.lower() or "design" in user_question.lower():
                return "The current documentation logs confirm that users highly praise the clean interface design layouts of the new smart display units."
            else:
                return f"[Mock Engine Response] Found {len(retrieved_chunks)} relevant items. Sample matched text text: '{retrieved_chunks[0][:60]}...'"

if __name__ == "__main__":
    print("🧪 Testing RAG Engine Standalone...")

    # Instantiate engine pointing to a test collection
    rag = RAGEngine(collection_name="test_tech_collection", chunk_size=300, chunk_overlap=30)

    # Attempt to pull real data from our ingestion file
    data_file = "data/raw_scraped/tech_gadgets_raw.csv"
    if os.path.exists(data_file):
        test_df = pd.read_csv(data_file)

        # Ingest the rows into the database
        rag.ingest_dataframe(test_df, text_column="text")

        # Ask a targeted question
        answer = rag.query("What are the core issues regarding battery performance?")
        print(f"\n🤖 Generated Answer:\n{answer}\n")
    else:
        print(f"❌ Missing {data_file}. Run ingestion first before executing test script.")