
### Updated `Readme.md`

📈 Live Stock Analysis Agent (Phase 2) — Project Summary & RAG Architecture

1.  Project Objective

This project is a multi-service application that fetches live stock prices (5-minute level), processes historical logs using Retrieval-Augmented Generation (RAG), and generates natural-language explanations using an LLM. It successfully integrates LLMs with real-world, time-series data to deliver insights and provide multi-factor alert notifications via a live web dashboard.

2.  Core Features

| Feature | Description |
| :--- | :--- |
| 🚨 **Multi-Factor Alert Engine** | Implements a powerful, five-part modular system that calculates real-time signals based on: Technical Indicators, Volume Metrics, Market Context (SPY, ^VIX), Calendar/Cyclical Patterns, and News Sentiment. |
| 🌐 **Resilient Data Pipeline** | Uses stable, small batches to fetch 5-minute live data for all tickers. Also features a **self-healing** 30-minute historical backfill that runs every 30 minutes to catch any missing data. |
| 🗃️ **Atomic & Orchestrated** | All data saves (CSV, JSON, FAISS) are **atomic** (using `os.replace`) to prevent read/write race conditions between services. The news fetch runs *serially* at the start of each cycle to guarantee sentiment data is fresh before alerts are run. |
| 🧠 **RAG Pipeline** | Implements a full RAG pipeline for time-series data, using historical stock logs as the knowledge base. The RAG index auto-reloads in the dashboard when it detects a newer file. |
| 💬 **LLM-Powered Insights** | Generates natural-language explanations of stock trends, anomalies, and movements. |
| 🖥️ **Stateless & Non-Blocking UI** | Features a **stateless** Dash UI (using `dcc.Store`) for visualization and scalability. The RAG chat is **non-blocking** (uses `background=True`) and displays Key Live Technical Indicators (RSI, VWAP) pre-calculated by the pipeline. |

3.  Technology Stack

| Category | Components |
| :--- | :--- |
| Data Fetching | yfinance, requests (for News API) |
| Data Handling | pandas, pandas-ta |
| Environment | python-dotenv, conda |
| Embeddings/Vector | sentence-transformers, FAISS |
| LLM Interface | Ollama (local) |
| Web/UI | dash, plotly, diskcache |
| Concurrency | subprocess, threading |

🚀 Getting Started

**Prerequisites**

  * **Conda:** Install the Conda package manager.
  * **Ollama:** Ensure the Ollama service is running locally, and the model specified in your `.env` file (`phi3:mini` by default) is downloaded.
  * **API Key (CRITICAL\!):** The pipeline requires an external key for news data.

<!-- end list -->

1.  Repository & Environment Setup

Create a file named `.gitignore` in the root directory:

```bash
# Python bytecode
__pycache__/
*.pyc

# Dash Diskcache Manager files
cache/

# Generated data, FAISS index, and JSON texts
data/

# Temporary files for atomic writes
*.tmp

# Local Environment Variables / Secrets
.env

# Conda environment directory
.venv/
venv/
```

Create a file named `.env` in the root directory to configure the pipeline (do NOT commit this file):

```env
# .env
# Stock tickers to monitor
TICKERS=AAPL,AMZN,GOOK,MSFT,TSLA

# Refresh interval in seconds (YFinance bulk fetch frequency)
REFRESH_INTERVAL=600

# Local data directory (use relative path for portability)
DATA_DIR=./data

# Log Level
LOG_LEVEL=INFO

# Ollama LLM model
OLLAMA_API_URL="http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME="phi3:mini"

# News/Sentiment API Configuration
NEWS_API_KEY=YOUR_NEWS_API_KEY
NEWS_REFRESH_INTERVAL=600 

# Data Retention
CSV_RETENTION_DAYS=30
```

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate stock_env
```

2.  Execution

Run the master launcher script. This single command handles the one-time historical data fetch (if needed) and starts all three services.

```bash
python run_all.py
```

**Note:** The first time you run this, it will check for 30-day historical data. If missing, it will fetch it in safe, small batches (with delays), which may take several minutes. Subsequent runs will start immediately. The pipeline runs an **immediate** first data fetch on startup, so data should appear within 1-2 minutes.

3.  Access the Dashboard

Once the services are running, open your web browser to:

👉 `http://127.0.0.1:8050`

4.  Shutdown

Press `Ctrl+C` in the terminal running `run_all.py`. The master script will gracefully terminate all three child processes.

⚙️ Project Architecture (File Breakdown)

The project is structured around three independent services managed by `run_all.py`.

**`environment.yml`**

  * **Purpose:** Defines the complete Conda environment named `stock_env`, listing all dependencies required for the pipeline (e.g., `yfinance`, `pandas-ta`) and RAG/Dashboard (e.g., `dash`, `faiss-cpu`, `sentence-transformers`).
  * **Key Detail:** Locks the Python version to 3.10.

**`run_all.py` (Master Launcher)**

  * **Purpose:** The single, intelligent entry point for the local environment. It manages the initial setup and life cycle of the three core services.
  * **Flow:**
    1.  **Historical Fetch:** Checks if `*_hist.csv` files exist. If not, runs a one-time, safe historical data fetch using a **batch size of 2** with delays.
    2.  **Service Launch:** Uses `subprocess` and `threading` to launch `phase2_pipeline.py`, `build_vector_index.py`, and `live_dashboard.py`.
    3.  **Shutdown:** Implements robust `SIGINT`/`SIGTERM` handling to ensure all child processes are terminated cleanly upon exit.

📊 Data Pipeline and Indexing

**`phase2_pipeline.py` (Data Ingestion Service)**

| Detail | Description |
| :--- | :--- |
| **Purpose** | The core data ingestion and alert orchestration service. |
| **Data Flow** | **Live (5-Min):** Runs every 10 minutes (600s). Fetches 1 day of 5-min data in stable **batches of 2** to avoid rate limits.<br>**Startup:** Runs an *immediate* first fetch, then begins the 10-minute wait cycle. |
| **Key Feature** | **Self-Healing Backfill:** Every 3rd run (\~30 mins), it also runs a "Smart Incremental Backfill" to heal the 30-min `_hist.csv` files, fetching any missing data. |
| **Key Components** | `LiveTracker` for managing in-memory data buffers.<br>`AlertEngine` orchestrates calls to the `signals/` directory.<br>`news_fetcher_task` runs **serially** at the start of each cycle to fetch sentiment data *before* alerts are calculated. |
| **Output** | Saves `_live.csv`, `_hist.csv`, and `live_alerts.json` (with pre-calculated TA) using **atomic writes** (`os.replace`) to prevent read/write conflicts. |

**`signals/` Directory (Modular Alert Logic)**

| File | Purpose |
| :--- | :--- |
| `technical.py` | Calculates all standard technical indicators (RSI, MACD, Bollinger Bands) from live data. |
| `volume.py` | Calculates volume-based signals (Volume Spikes, VWAP Crossovers, Divergence). |
| `market_context.py`| Compares stock performance to market indices (SPY) and checks risk sentiment (^VIX). |
| `calendar.py` | Generates time-based alerts (Market Open/Close volatility, Day of Week context). |
| `sentiment.py` | Manages the API call and caching for external news headlines and sentiment analysis. |

**`build_vector_index.py` (RAG Indexing Service)**

| Detail | Description |
| :--- | :--- |
| **Purpose** | Builds and maintains the vector index for the LLM Chatbot's knowledge base. |
| **Flow** | Runs on a loop, rebuilding the entire index every 15 minutes. Calculates Technical Indicators (MA/RSI). Generates "Smart Chunks" of text, including facts, analysis, and an Engaging Hook. |
| **Key Components** | `SentenceTransformer` for embeddings. `faiss` for vector search. `pandas-ta` for technical analysis. Indexes all tickers including SPY and ^VIX for full context. |
| **Local Limitation** | Inefficient full index rebuild process and dependency on local file I/O for the index files. Becomes slow as historical data grows. |

💻 Dashboard and Chatbot

**`live_dashboard.py` (Web UI/API Gateway)**

| Detail | Description |
| :--- | :--- |
| **Purpose** | The user-facing web application that displays real-time data and hosts the interactive RAG-enabled LLM Chatbot. |
| **Visualization** | Displays Dynamic Historical charts, Live Price/Volume charts, a Key Live Indicators panel (RSI, VWAP, etc.), and real-time alert logs. |
| **State Mgt.** | **Stateless Design:** Uses `dcc.Store` components to hold all data. This makes the dashboard stateless and process-safe for production servers (Gunicorn). |
| **RAG Chatbot Logic** | Uses `search_rag_index` to retrieve semantic context. This function is **self-healing** and automatically reloads the RAG index if it detects a newer version on disk. It sends history and context to the Ollama LLM using a strict, 17-rule system prompt. |
| **Performance** | **FIXED:** The `chat_llm` callback runs as a Dash **background callback** (`background=True`), preventing the UI from freezing while the local LLM generates a response. |

💡 LLM & RAG Capabilities

The agent's "intelligence" comes from combining both historical and real-time data to answer user queries.

  * **LLM Explanations:** The LLM uses RAG to generate natural-language insights. It combines context from the historical 30-min data (via FAISS) and the live 5-min data (via the background pipeline).
  * **Synthesis Focus:** The LLM is specifically prompted to synthesize signals from all five modular categories (Technical, Volume, Market Context, Sentiment, Calendar) for richer, grounded explanations.
  * **Example Prompts:** The system can successfully answer prompts such as:
      * *"Explain today’s price movement for {ticker}."*
      * *"Summarize last week’s stock trend."*
      * *"What is the current market sentiment based on VIX?"*

**User-facing Deliverables:**

  * **Dashboard:** An interactive dashboard displays current prices, historical trends, the LLM-generated explanations, and live alerts.
  * **Insights:** The system delivers volatility analysis, trend patterns, and anomaly detection.

☁️ Future Optimization: Cloud Deployment & Microservices

The current single-machine architecture can be refactored into three distinct Microservices for improved scalability, resilience, and independent deployment in a cloud environment (e.g., AWS, GCP).

**I. Refactoring for Microservices**

| Local Service | Cloud Microservice Name | Core Optimizations / Cloud Service |
| :--- | :--- | :--- |
| `phase2_pipeline.py` | **Data Ingestion Service (DIS)** | Decouple I/O using Message Queues (Kafka/PubSub). Migrate to a Time-Series Database (TimescaleDB). |
| `build_vector_index.py` | **RAG Indexing Service (RIS)** | Migrate to a Cloud-Native Vector Database (e.g., Pinecone). Implement Incremental Indexing. Deploy as a Serverless Function (Lambda/Cloud Functions). |
| `live_dashboard.py` | **Dashboard/API Gateway Service** | Deploy on Cloud Run/App Service using a production server (Gunicorn/Uvicorn). Communicate with the LLM via a dedicated internal endpoint. |

**II. Cloud Deployment Strategy**

  * **Containerization:** Use Docker to create separate images for the DIS, RIS, and Dashboard services.
  * **Orchestration:** Use Kubernetes (K8s) or Cloud Run for scalable process management.
  * **Data Persistence:** Migrate all stock data from local CSVs to a managed time-series database. Move the RAG index to a cloud-native vector database.
  * **LLM Infrastructure:** Deploy Ollama as a dedicated, GPU-accelerated service or use a managed LLM provider for fast, scalable inference.
  * **API Communication:** Implement gRPC for low-latency, internal service-to-service communication between the microservices.

📚 References

  * yfinance: `https://pypi.org/project/yfinance/`
  * Dash: `https://dash.plotly.com/`
  * pandas-ta: `https://github.com/twopirllc/pandas-ta`
  * Ollama: `https://ollama.com/`
  * FAISS: `https://faiss.ai/`
  * sentence-transformers: `https://www.sbert.net/`
  * Dash Diskcache: `https://dash.plotly.com/background-callbacks`