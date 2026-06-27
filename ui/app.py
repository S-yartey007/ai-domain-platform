import os
import streamlit as st
import pandas as pd
import sys
from src.utils.config_loader import load_config
from src.engines.topic_tracker import TopicTrackerEngine
from src.engines.rag_engine import RAGEngine


# Set up page configurations
st.set_page_config(page_title="Modular GenAI Platform", layout="wide", page_icon="🚀")

st.title("💡 Cross-Domain Market Intelligence Platform")
st.caption("A modular framework blending traditional NLP clustering with Retrieval-Augmented Generation (RAG).")

# 1. Sidebar - Configuration & Controls
st.sidebar.header("🎯 Project Controller")

# Load configuration to display info
try:
    config = load_config()
    domain_name = config.get("name", "Unknown Domain")
    domain_key = config.get("domain_key", "unknown")

    st.sidebar.success(f"**Active Vertical:**\n{domain_name}")
    st.sidebar.info(f"**Config Context:** `{domain_key}`")
except Exception as e:
    st.sidebar.error(f"Error loading config.yaml: {e}")
    st.stop()

# Define paths based on the active domain
data_path = f"data/raw_scraped/{domain_key}_raw.csv"

# Check if data exists
if not os.path.exists(data_path):
    st.warning(f"⚠️ Data file not found for this domain: `{data_path}`. Please run data ingestion first using your terminal: `python -m src.run_ingestion`")
    st.stop()

# Load the raw dataset
df_raw = pd.read_csv(data_path)

# 2. Main Dashboard Tab Layout
tab1, tab2, tab3 = st.tabs(["📊 Sentiment & Topic Clusters", "💬 RAG Knowledge Bot", "📂 Source Data View"])

# --- TAB 1: TOPIC CLUSTERING ---
with tab1:
    st.header("🧠 Emerging Topic Modeling")
    st.write("This engine vectorizes public sentiment data and uses Unsupervised K-Means clustering combined with an LLM layer to discover hidden trends.")

    if st.button("🔮 Run Live Clustering Engine", type="primary"):
        with st.spinner("Analyzing text patterns and generating AI cluster labels..."):
            try:
                prompt = config.get("topic_modeling", {}).get("llm_label_prompt", "")
                min_clusters = config.get("topic_modeling", {}).get("min_cluster_size", 3)

                # Run the backend clustering engine
                tracker = TopicTrackerEngine(llm_prompt=prompt, min_clusters=3)
                df_clustered = tracker.cluster_and_label(data_path)

                # Show aggregate visualization
                st.subheader("📍 Discovered Key Themes")
                topic_counts = df_clustered['assigned_topic'].value_counts()
                st.bar_chart(topic_counts)

                # Show sample records from clusters
                st.subheader("📋 Contextual Snippets per Topic")
                for topic in topic_counts.index:
                    with st.expander(f"Topic Focus: {topic}"):
                        samples = df_clustered[df_clustered['assigned_topic'] == topic]['text'].head(3)
                        for sample in samples:
                            st.write(f"- {sample}")
            except Exception as e:
                st.error(f"Clustering process encountered an issue: {e}")

# --- TAB 2: RAG INTERACTIVE CHAT ---
with tab2:
    st.header("💬 Context-Grounded Document Expert")
    st.write("Ask questions directly against your ingestion database. This RAG engine searches semantic indices to synthesize response context, heavily curbing hallucinations.")

    # Initialize RAG Engine state
    collection_name = config.get("rag", {}).get("collection_name", "default_collection")
    chunk_size = config.get("rag", {}).get("chunk_size", 500)
    chunk_overlap = config.get("rag", {}).get("chunk_overlap", 50)

    rag = RAGEngine(collection_name=collection_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Simple setup state indicator for database index tracking
    if "rag_indexed" not in st.session_state:
        st.session_state["rag_indexed"] = False

    if not st.session_state["rag_indexed"]:
        if st.button("🗂️ Index Collection in Vector Store"):
            with st.spinner("Building local mathematical embedding database matrices..."):
                rag.ingest_dataframe(df_raw, text_column="text")
                st.session_state["rag_indexed"] = True
                st.success("Vector DB Index built successfully! Proceed to type queries below.")

    if st.session_state["rag_indexed"]:
        user_query = st.text_input("💬 Enter an analysis question about the dataset:")
        if user_query:
            with st.spinner("Searching matching coordinates and reasoning answer..."):
                answer = rag.query(user_query)
                st.markdown("### 🤖 Engine Response:")
                st.info(answer)

# --- TAB 3: SOURCE VIEWER ---
with tab3:
    st.header("📂 Raw Ingested Feed Frame")
    st.write(f"Displaying raw unstructured data loaded from `{data_path}`")
    st.dataframe(df_raw, use_container_width=True)