Here is your updated `Readme.md` file. I have added a new **"🔗 Project Resources"** section containing your Video and Medium links just before the References section, which is a standard location for external project links.

````markdown
## 1. Project Objective

This project is a multi-service application that fetches live stock prices (5-minute level), processes historical logs using Retrieval-Augmented Generation (RAG), and generates natural-language explanations using an LLM. It successfully integrates LLMs with real-world, time-series data to deliver insights and provide multi-factor alert notifications via a live web dashboard.

## 2. Core Features

| Feature | Description |
| :--- | :--- |
| 🚨 **Multi-Factor Alert Engine** | Implements a powerful, five-part modular system that calculates real-time signals based on: Technical Indicators, Volume Metrics, Market Context (SPY, ^VIX), Calendar/Cyclical Patterns, and News Sentiment. |
| 🌐 **Resilient Data Pipeline** | Uses a **Single Bulk API Call** every 5 minutes to maintain data snapshot consistency. Also features a **self-healing** 30-minute historical backfill that runs every 15 minutes to catch any missing data. |
| 🗃️ **Atomic & Orchestrated** | All data saves (CSV, JSON, FAISS) are **atomic** (using `os.replace`) to prevent read/write race conditions between services. The news fetch runs *serially* at the start of each cycle to guarantee sentiment data is fresh before alerts are run. |
| 🧠 **RAG Pipeline** | Implements a full RAG pipeline for time-series data, using historical stock logs as the knowledge base. The RAG index auto-reloads in the dashboard when it detects a newer file. |
| 💬 **LLM-Powered Insights** | Generates natural-language explanations of stock trends, anomalies, and movements. |
| 🖥️ **Stateless & Non-Blocking UI** | Features a **stateless** Dash UI (using `dcc.Store`) for visualization and scalability. The RAG chat is **non-blocking** (uses `background=True`) and displays Key Live Technical Indicators (RSI, VWAP) pre-calculated by the pipeline. |

## 3. Technology Stack

| Category | Components |
| :--- | :--- |
| **Data Storage Point** | **Local `data/` directory** (CSVs, JSONs, FAISS Index) |
| Data Fetching | yfinance, requests (for News API) |
| Data Handling | pandas, pandas-ta |
| Environment | python-dotenv, conda |
| Embeddings/Vector | sentence-transformers, FAISS |
| LLM Interface | Ollama (local) |
| Web/UI | dash, plotly, diskcache |
| Concurrency | subprocess, threading |

---

## 🚀 Getting Started

### Prerequisites

* **Conda:** Install the Conda package manager.
* **Ollama:** Ensure the Ollama service is running locally, and the model specified in your `.env` file (`phi3:mini` by default) is downloaded.
* **API Key (CRITICAL!):** The pipeline requires an external key for news data.

### 1. Repository & Environment Setup

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
````

Create a file named `.env` in the root directory to configure the pipeline (do NOT commit this file):

```env
# .env
# Stock tickers to monitor
TICKERS=AAPL,AMZN,GOOGL,MSFT,TSLA

# --- Pipeline Configuration ---
# Controls the strict single-call frequency (5 minutes)
REFRESH_INTERVAL=300

# Local data directory (use relative path for portability)
DATA_DIR=./data

# Log Level
LOG_LEVEL=INFO

# Ticker batch size for HISTORICAL SETUP (Default 8, effectively hardcoded to 2 in run_all.py)
TICKER_BATCH_SIZE=8 

# --- LLM Configuration ---
OLLAMA_API_URL="http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME="phi3:mini"
OLLAMA_TIMEOUT=120 

# --- News/Sentiment API Configuration ---
NEWS_API_KEY=YOUR_NEWS_API_KEY
NEWS_REFRESH_INTERVAL=600 

# --- Rate Limit Defense ---
# 30-minute stall if the single bulk request fails
EMERGENCY_COOLDOWN=1800 

# Data Retention
CSV_RETENTION_DAYS=30
```

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate stock_env
```

### 2\. Execution

Run the master launcher script. This single command handles the one-time initial data fetch (if needed) and starts all three services.

```bash
python run_all.py
```

**Note on Initial Data Fetch (CRITICAL UPDATE):**

The first time you run this, the system will initiate an **Ultra-Safe Strict Serial Fetch** for 30 days of 5-minute data for each of your tickers.

  * **Strict Retry Logic:** To prevent incomplete data, the setup enters a blocking loop. If any ticker fails to download (e.g., due to rate limits), it will **automatically retry** indefinitely (or up to a max retry limit) until the file is successfully created.
  * **Rate Limiting:** It enforces a **5-minute delay** between fetching each individual ticker.
  * **Total Setup Time:** This initial fetch will take approximately **35-40 minutes** to complete.
  * **Final Cooldown:** Once all data is secured, the system waits a final 5 minutes before launching the dashboard to ensure API safety.
  * *Subsequent runs will skip this initial fetch entirely if the data files exist.*

**Approximate Initialization Time (No Data):**

| Phase | Time | Details |
| :--- | :--- | :--- |
| **Initial Serial Data Fetch** | $\approx$ **35 - 40 minutes** | Fetching 8 tickers with mandatory 5-minute cooldowns between batches. |
| **System Cooldown** | **5 minutes** | Mandatory pause after the final historical fetch batch before launching services. |
| **Dashboard/RAG Ready** | **Immediate** | Dashboard loads historical charts, RAG index, and initial alerts right away. |
| **First Live Fetch Cycle** | **5 minutes** | The pipeline's first full cycle, which updates and appends the latest 5-minute bar to the unified file. |
| **Total Time to Full Data** | $\approx$ **45 - 50 minutes** | Time until the dashboard is fully populated with live data and a functioning chatbot. |

### 3\. Access the Dashboard

Once the services are running, open your web browser to:

👉 `http://127.0.0.1:8050`

> **🕒 Time Zone Visualization Note:**
> The stock data is processed and stored internally in **UTC** to maintain consistency across all services. Consequently, **the charts on the dashboard display timestamps in UTC.**
> Users in other time zones (e.g., CST/EST) should account for the time difference when viewing live candles.
>
>   * *Example:* Market Open (09:30 EST / 08:30 CST) = **14:30 UTC**.

### 4\. Shutdown

Press `Ctrl+C` in the terminal running `run_all.py`. The master script will gracefully terminate all three child processes.

-----

## ⚙️ Project Architecture (File Breakdown)

The project is structured around three independent services managed by `run_all.py`.

**`environment.yml`**

  * **Purpose:** Defines the complete Conda environment named `stock_env`, listing all dependencies required for the pipeline (e.g., `yfinance`, `pandas-ta`) and RAG/Dashboard (e.g., `dash`, `faiss-cpu`, `sentence-transformers`).
  * **Key Detail:** Locks the Python version to 3.10.

**`run_all.py` (Master Launcher)**

  * **Purpose:** The single, intelligent entry point for the local environment. It manages the initial setup and life cycle of the three core services.
  * **Flow:**
    1.  **Strict Initial Fetch:** Checks if the unified `<TICKER>.csv` file exists. If not, runs a **SERIALLY THROTTLED FETCH** for 30 days of 5-minute data. It includes a **retry loop** that prevents the dashboard from starting until *all* data files are verified.
    2.  **Service Launch:** Uses `subprocess` and `threading` to launch `phase2_pipeline.py`, `build_vector_index.py`, and `live_dashboard.py`.
    3.  **Shutdown:** Implements robust `SIGINT`/`SIGTERM` handling to ensure all child processes are terminated cleanly upon exit.

### 📊 Data Pipeline and Indexing

**`phase2_pipeline.py` (Data Ingestion Service)**

| Detail | Description |
| :--- | :--- |
| **Purpose** | The core data ingestion and alert orchestration service. |
| **Data Flow** | **Live (5-Min):** Runs every 5 minutes (300s). Performs a **SINGLE BULK API CALL** for all tickers to ensure data consistency and minimize request frequency. |
| **Key Feature** | **Incremental Update:** The live fetch logic automatically identifies the last timestamp in the unified `<TICKER>.csv` file and **appends only the missing 5-minute bars**, maintaining data integrity. |
| **Adaptive Cooldown** | If the single bulk fetch fails, the system stalls the next cycle for 30 minutes to reset the aggressive API block. |
| **Output** | Saves `<TICKER>.csv` (unified data) and `live_alerts.json` (with pre-calculated TA) using **atomic writes** (`os.replace`) to prevent read/write conflicts. |

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
| **Flow** | Reads the unified `<TICKER>.csv` file containing 5-minute data. It then **resamples this data to 30-minute and daily intervals** to generate historical context RAG chunks. |
| **Key Components** | `SentenceTransformer` for embeddings. `faiss` for vector search. `pandas-ta` for technical analysis. Indexes all tickers including SPY and ^VIX for full context. |
| **Local Limitation** | Inefficient full index rebuild process and dependency on local file I/O for the index files. Becomes slow as historical data grows. |

### 💻 Dashboard and Chatbot

**`live_dashboard.py` (Web UI/API Gateway)**

| Detail | Description |
| :--- | :--- |
| **Purpose** | The user-facing web application that displays real-time data and hosts the interactive RAG-enabled LLM Chatbot. |
| **Visualization** | Displays Dynamic Historical charts, Live Price/Volume charts, a Key Live Indicators panel (RSI, VWAP, etc.), and real-time alert logs. |
| **State Mgt.** | **Stateless Design:** Uses `dcc.Store` components to hold all data. It loads the single `<TICKER>.csv` and uses **time filtering** to render the different charts. |
| **RAG Chatbot Logic** | Uses `search_rag_index` to retrieve semantic context. This function is **self-healing** and automatically reloads the RAG index if it detects a newer version on disk. It sends history and context to the Ollama LLM using a strict, 17-rule system prompt. |
| **Performance** | **FIXED:** The `chat_llm` callback runs as a Dash **background callback** (`background=True`), preventing the UI from freezing while the local LLM generates a response. |

## 💡 LLM & RAG Capabilities

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

## ☁️ Future Optimization: Cloud Deployment & Microservices

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

## 🔗 Project Resources

  * **▶️ Live Demo:** [Watch on Google Drive](https://drive.google.com/file/d/17CM8klaQCYLM_1woCKoZp3vZIHidr8a6/view?usp=drive_link)
  * **✍️ Medium Article:** [Stop Staring at Charts: Building a Real-Time AI Financial Analyst](https://medium.com/@s.parshwa18/stop-staring-at-charts-building-a-real-time-ai-financial-analyst-with-rag-and-quadratic-context-4de44e67b286?postPublishedType=repub)

### 📚 References

  * yfinance: `https://pypi.org/project/yfinance/`
  * Dash: `https://dash.plotly.com/`
  * pandas-ta: `https://github.com/twopirllc/pandas-ta`
  * Ollama: `https://ollama.com/`
  * FAISS: `https://faiss.ai/`
  * sentence-transformers: `https://www.sbert.net/`
  * Dash Diskcache: `https://dash.plotly.com/background-callbacks`

<!-- end list -->

```
```