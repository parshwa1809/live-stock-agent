# 📈 Live Stock Analysis Agent — Project Summary & RAG Architecture

### 1\. Project Objective

This project is a terminal-based agent that fetches live stock prices (minute-level), processes historical logs using Retrieval-Augmented Generation (RAG), and generates natural-language explanations using an LLM. It successfully integrates LLMs with real-world, time-series data to deliver insights and provide real-time alert notifications for stock movements.

-----

### 2\. Core Features

  * **RAG Pipeline:** Implements a full RAG pipeline for time-series data, using historical stock logs as the knowledge base.
  * **Data Ingestion:** Ingests both live (1-minute) and historical (30-minute interval) stock data from Yahoo Finance.
  * **Vector Embeddings:** Creates and manages a FAISS index for efficient semantic retrieval over historical data.
  * **LLM-Powered Insights:** Generates natural-language explanations of stock trends, anomalies, and movements.
  * **Live Dashboard:** Features a local UI (Dash/FastAPI) for visualization and interactive insights.
  * **Real-time Alerts:** Provides threshold-based alerts for multiple tickers.

-----

### 3\. Technology Stack

  * **Data Fetching:** `yfinance`
  * **Data Handling:** `pandas`, `pandas-ta`
  * **Environment:** `python-dotenv`, `conda`
  * **Embeddings:** `sentence-transformers`
  * **Vector Index:** `FAISS`
  * **LLM:** `Ollama` (local), `OpenAI API` (optional)
  * **Backend/Dashboard:** `dash`, `dash-bootstrap-components`, `FastAPI`
  * **Concurrency:** `subprocess`, `threading`

-----

## 🚀 Getting Started

### Prerequisites

1.  **Conda:** Install the Conda package manager.
2.  **Ollama:** Ensure the **Ollama service** is running locally on your machine, and the model specified in your `.env` file (`phi3:mini` by default) is downloaded.
3.  **Repository Setup:** Ensure you have created your `.gitignore` file (see below) and pushed all necessary source files to your repository.

### 1\. Repository & Environment Setup

Create a file named **`.gitignore`** in the root directory:

```gitignore
# Python bytecode
__pycache__/
*.pyc

# Dash Diskcache Manager files
cache/

# Generated data, FAISS index, and JSON texts
Data/

# Local Environment Variables / Secrets
.env

# Conda environment directory
.venv/
venv/
```

Create a file named **`.env`** in the root directory to configure the pipeline (do NOT commit this file):

```env
# .env
TICKERS="AAPL,AMZN,GOOGL,MSFT,TSLA"
REFRESH_INTERVAL=300
DATA_DIR="./Data"
# Ensure Ollama is running and accessible
OLLAMA_API_URL="http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME="phi3:mini"
```

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate stock_env
```

### 2\. Execution

Run the master launcher script to start all three services concurrently:

```bash
python run_all.py
```

### 3\. Access the Dashboard

Once the services are running, open your web browser to:

👉 **[http://127.0.0.1:8050](http://127.0.0.1:8050)**

### 4\. Shutdown

Press **Ctrl+C** in the terminal running `run_all.py`. The master script will gracefully terminate all three child processes.

-----

## ⚙️ Project Architecture (File Breakdown)

The project is structured around three independent services managed by `run_all.py`.

### `environment.yml`

  * **Purpose:** Defines the complete Conda environment named `stock_env`, listing all dependencies required for the pipeline (e.g., `yfinance`, `pandas-ta`) and RAG/Dashboard (e.g., `dash`, `faiss-cpu`, `sentence-transformers`).
  * **Key Detail:** Locks the Python version to `3.10`.

### `run_all.py` (Master Launcher)

  * **Purpose:** The single entry point for the local environment. It manages the life cycle of the three core services.
  * **Flow:** Uses Python's `subprocess` and `threading` to launch:
    1.  `phase2_pipeline.py` (Data Ingestion)
    2.  `build_vector_index.py` (RAG Indexing)
    3.  `live_dashboard.py` (Web UI)
  * **Shutdown:** Implements robust `SIGINT`/`SIGTERM` handling to ensure all child processes are terminated cleanly upon exit.

-----

## 📊 Data Pipeline and Indexing

### `phase2_pipeline.py` (Data Ingestion Service)

| Detail | Description |
| :--- | :--- |
| **Purpose** | The core data ingestion and processing script. |
| **Data Flow** | **Historical:** The *unreliable initial fetch* is **bypassed** (`initial_setup()` is commented out) to maintain stability. Data is assumed to exist. **Live:** Polls **1-minute** data for the current day (`*_live.csv`) every **5 minutes** (300s). |
| **Key Components** | `LiveTracker`/`DataManager` for data persistence. **Stable Logic:** Uses a **5-minute refresh cycle** and a **10-second delay** between individual ticker fetches for reliability. `AlertEngine` for rule-based checks (Trend, Price Swing, Volume). `LLMInterface` for synchronous Ollama communication. |
| **Local Limitation** | Heavy reliance on `time.sleep()` and local CSV files (`./Data`). The LLM interface is synchronous, blocking the thread while waiting for Ollama. |

### `build_vector_index.py` (RAG Indexing Service)

| Detail | Description |
| :--- | :--- |
| **Purpose** | Builds and maintains the vector index for the LLM Chatbot's knowledge base. |
| **Flow** | Runs on a loop, rebuilding the entire index every **15 minutes**. Calculates **Technical Indicators (MA/RSI)**. Generates **"Smart Chunks"** of text, including facts, analysis, and an **Engaging Hook**. Embeds chunks and stores them in **FAISS** and JSON. |
| **Key Components** | `SentenceTransformer` for embeddings. `faiss` for vector search. `pandas-ta` for technical analysis. |
| **Local Limitation** | Inefficient full index rebuild process and dependency on local file I/O for the index files. |

-----

## 💻 Dashboard and Chatbot

### `live_dashboard.py` (Web UI/API Gateway)

| Detail | Description |
| :--- | :--- |
| **Purpose** | The user-facing web application that displays real-time data and hosts the interactive **RAG-enabled LLM Chatbot**. |
| **Visualization** | Displays two pairs of charts (**Dynamic Historical** chart updates its endpoint with the latest live data point, and Live Price/Volume) and real-time alerts. |
| **RAG Chatbot Logic** | Uses **`search_rag_index`** to retrieve semantic context. Sends the chat history and retrieved context to the **Ollama** LLM using a **strict system prompt (17 rules)** to ensure a professional, data-driven response and eliminate repetition. |
| **Local Limitation** | The `chat_llm` callback runs **synchronously** on the main thread, causing the UI to freeze momentarily during the LLM's response generation. |

-----

## 💡 LLM & RAG Capabilities

The agent's "intelligence" comes from combining both historical and real-time data to answer user queries.

  * **LLM Explanations:** The LLM uses RAG to generate natural-language insights. It combines context from the **historical 30-min data** (via FAISS) and the **recent 1-min in-memory data**.
  * **Example Prompts:** The system can successfully answer prompts such as:
      * "Explain today’s price movement for {ticker}."
      * "Summarize last week’s stock trend."
      * "Identify potential anomalies in recent data."
  * **User-facing Deliverables:**
      * **Dashboard:** An interactive dashboard displays current prices, historical trends, the LLM-generated explanations, and live alerts.
      * **Insights:** The system delivers volatility analysis, trend patterns, and anomaly detection.

-----

## ☁️ Future Optimization: Cloud Deployment & Microservices

The current single-machine architecture can be refactored into three distinct **Microservices** for improved scalability, resilience, and independent deployment in a cloud environment (e.g., AWS, GCP).

### I. Refactoring for Microservices

| Local Service | Cloud Microservice Name | Core Optimizations / Cloud Service |
| :--- | :--- | :--- |
| `phase2_pipeline.py` | **Data Ingestion Service (DIS)** | Decouple I/O using **Message Queues (Kafka/PubSub)**. Migrate to a **Time-Series Database (TimescaleDB)**. |
| `build_vector_index.py` | **RAG Indexing Service (RIS)** | Migrate to a **Cloud-Native Vector Database (e.g., Pinecone)**. Implement **Incremental Indexing**. Deploy as a **Serverless Function (Lambda/Cloud Functions)**. |
| `live_dashboard.py` | **Dashboard/API Gateway Service** | Deploy on **Cloud Run/App Service** using a production server (**Gunicorn/Uvicorn**). Communicate with the LLM via a dedicated internal endpoint. |

### II. Cloud Deployment Strategy

1.  **Containerization:** Use **Docker** to create separate images for the DIS, RIS, and Dashboard services.
2.  **Orchestration:** Use **Kubernetes (K8s)** or **Cloud Run** for scalable process management.
3.  **Data Persistence:** Migrate all stock data from local CSVs to a **managed time-series database**. Move the RAG index to a **cloud-native vector database**.
4.  **LLM Infrastructure:** Deploy **Ollama** as a dedicated, **GPU-accelerated** service or use a managed LLM provider for fast, scalable inference.
5.  **API Communication:** Implement **gRPC** for low-latency, internal service-to-service communication between the microservices.

-----

## 📚 References

  * **yfinance:** [https://pypi.org/project/yfinance/](https://pypi.org/project/yfinance/)
  * **FastAPI:** [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
  * **Chart.js:** [https://www.chartjs.org/docs/latest/](https://www.chartjs.org/docs/latest/)
  * **OpenAI API:** [https://platform.openai.com/docs/](https://platform.openai.com/docs/)
  * **FAISS:** [https://faiss.ai/](https://faiss.ai/)
  * **sentence-transformers:** [https://www.sbert.net/](https://www.sbert.net/)