Here is the complete `README.md` file. I have applied **hard wrapping** to the text sections so they fit within a readable "A4 width" when viewing the raw code, while still rendering correctly as full paragraphs on GitHub.

````markdown
# Live Stock Analysis Agent & Agentic RAG

## 1. Project Objective
This project is a multi-service application that fetches live stock prices (5-minute level), 
processes historical logs using **Retrieval-Augmented Generation (RAG)**, and generates 
natural-language explanations using an LLM. 

It successfully integrates LLMs with real-world, time-series data to deliver insights 
and provide multi-factor alert notifications via a live web dashboard.

---

## 2. Core Features
* **🚨 Multi-Factor Alert Engine:** Calculates real-time signals based on Technical Indicators, 
  Volume Metrics, Market Context (SPY, ^VIX), Calendar/Cyclical Patterns, and News Sentiment.
* **🌐 Resilient Data Pipeline:** Uses a **Single Bulk API Call** every 5 minutes to maintain 
  consistency. Features a **self-healing** 30-minute backfill mechanism to catch missing data.
* **🗃️ Atomic & Orchestrated:** Uses `os.replace` for atomic data saves to prevent race 
  conditions. News fetching runs serially to guarantee fresh sentiment data before alerts trigger.
* **🧠 RAG Pipeline:** Indexes historical stock logs as a knowledge base. The index auto-reloads 
  dynamically when new data is detected.
* **💬 LLM-Powered Insights:** Generates natural-language explanations of trends, anomalies, 
  and movements using the retrieved context.
* **🖥️ Stateless & Non-Blocking UI:** A stateless Dash UI (via `dcc.Store`) with a background 
  process for the RAG chat to ensure responsiveness.

---

## 3. Technology Stack
* **Core Env:** `python-dotenv`, `conda`, `subprocess`, `threading`
* **Data Handling:** `pandas`, `pandas-ta`, `yfinance`, `requests`
* **AI/ML:** `Ollama` (local), `sentence-transformers`, `FAISS`, `RAG`
* **Web/UI:** `Dash`, `Plotly`, `diskcache`
* **Storage:** Local `data/` directory (CSVs, JSONs, FAISS Index)

---

## 4. Project Architecture

The project is structured around three independent services managed by a master launcher.

### 🟢 Master Launcher: `run_all.py`
The single entry point for the environment.
* **Strict Initial Fetch:** Checks for data; if missing, runs a **serially throttled fetch** for 30 days of data (with retry logic) to ensure a complete dataset.
* **Service Launch:** Uses `subprocess` to launch the Pipeline, Indexer, and Dashboard.
* **Shutdown:** Handles `SIGINT` to cleanly terminate all child processes.

### 🟡 Data Ingestion: `phase2_pipeline.py`
The core service for data and alerts (Runs every 5 mins).
* **Mechanism:** Performs a **Single Bulk API Call** for all tickers to ensure data consistency.
* **Incremental Update:** Appends only missing 5-minute bars to the unified `<TICKER>.csv`.
* **Safety:** Stalls for 30 minutes if the bulk fetch fails to reset aggressive API limits.

### 🟣 RAG Indexing: `build_vector_index.py`
Builds the knowledge base for the Chatbot.
* **Process:** Resamples 5-min data to 30-min and Daily intervals to generate 
  meaningful vector embeddings for the LLM.
* **Context:** Indexes SPY and ^VIX alongside tickers for broader market context.

### 🔵 Dashboard & Chatbot: `live_dashboard.py`
The user-facing Web UI.
* **Visuals:** Dynamic Historical charts, Live Price/Volume, and Key Live Indicators (RSI, VWAP).
* **RAG Logic:** Uses `search_rag_index` to retrieve semantic context. Auto-reloads index on file change.
* **Performance:** The Chatbot runs as a **background callback**, preventing UI freezes 
  while generating answers.

> **📂 `signals/` Directory:** Contains modular logic for `technical.py` (RSI/MACD), 
> `volume.py`, `market_context.py`, `calendar.py`, and `sentiment.py`.

---

## 5. Getting Started

### Prerequisites
1.  **Conda** (Package Management)
2.  **Ollama** (Must be running locally with `phi3:mini` model recommended)
3.  **News API Key** (Required for sentiment analysis)

### Step 1: Environment Setup

**1. Create `.gitignore`**
```bash
__pycache__/
*.pyc
cache/
data/
*.tmp
.env
.venv/
venv/
````

**2. Create `.env` file**

```env
TICKERS=AAPL,AMZN,GOOGL,MSFT,TSLA
REFRESH_INTERVAL=300
DATA_DIR=./data
LOG_LEVEL=INFO
TICKER_BATCH_SIZE=8 

# LLM Config
OLLAMA_API_URL="http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME="phi3:mini"
OLLAMA_TIMEOUT=120 

# API Keys
NEWS_API_KEY=YOUR_NEWS_API_KEY
NEWS_REFRESH_INTERVAL=600 

# Safety
EMERGENCY_COOLDOWN=1800 
CSV_RETENTION_DAYS=30
```

**3. Install Dependencies**

```bash
conda env create -f environment.yml
conda activate stock_env
```

### Step 2: Execution

Run the master script:

```bash
python run_all.py
```

> **⚠️ Critical Note on First Run:**
> The system will perform an **Ultra-Safe Serial Fetch** (fetching 30 days of data for
> 8 tickers one by one).
>
>   * It waits **5 minutes** between each ticker to prevent rate limits.
>   * **Total Wait Time:** \~35-40 minutes.
>   * Subsequent runs will skip this step if data exists.

### Step 3: Access Dashboard

Open your browser to: **`http://127.0.0.1:8050`**

  * **Time Zone Note:** All data is stored and displayed in **UTC**.
    (e.g., Market Open 09:30 EST = 14:30 UTC).

-----

## 6\. LLM & RAG Capabilities

The agent combines historical context with real-time data to answer queries like
*"Explain today’s price movement for AAPL"* or *"What is the market sentiment?"*

1.  **Synthesis:** The LLM is prompted to synthesize signals from all five modules
    (Technical, Volume, Context, Sentiment, Calendar).
2.  **Grounding:** Responses are grounded in the 30-minute historical chunks retrieved
    via FAISS to reduce hallucinations.

-----

## 7\. Future Optimization: Cloud Deployment

The architecture is designed to be split into Microservices for cloud deployment (AWS/GCP).

  * **Data Ingestion:** Decouple using Kafka; migrate CSVs to TimescaleDB.
  * **RAG Indexing:** Migrate FAISS to a vector DB (Pinecone); run on Serverless functions.
  * **Dashboard:** Deploy on Cloud Run using Gunicorn.

-----

## 🔗 Project Resources

  * **▶️ Live Demo:** [Watch on Google Drive](https://drive.google.com/file/d/17CM8klaQCYLM_1woCKoZp3vZIHidr8a6/view?usp=drive_link)
  * **✍️ Medium Article:** [Stop Staring at Charts: Building a Real-Time AI Financial Analyst](https://medium.com/@s.parshwa18/stop-staring-at-charts-building-a-real-time-ai-financial-analyst-with-rag-and-quadratic-context-4de44e67b286?postPublishedType=repub)

### 📚 References

  * `yfinance`: https://pypi.org/project/yfinance/
  * `Dash`: https://dash.plotly.com/
  * `pandas-ta`: https://github.com/twopirllc/pandas-ta
  * `Ollama`: https://ollama.com/
  * `FAISS`: https://faiss.ai/

<!-- end list -->

```
```
