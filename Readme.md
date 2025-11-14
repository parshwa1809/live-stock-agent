I
# 📈 Live Stock Analysis Agent (Phase 2)

This project establishes a robust, real-time stock analysis system. Designed for a **Local Development Environment**, it runs three concurrent services: a data ingestion pipeline, a RAG vector indexing service, and a live web dashboard featuring an LLM-powered financial analyst chat.

---

## 🚀 Getting Started

### Prerequisites

1.  **Conda:** Install the Conda package manager.
2.  **Ollama:** Ensure the **Ollama service** is running locally on your machine, and the model specified in your `.env` file (`phi3:mini` by default) is downloaded.
3.  **Repository Setup:** Ensure you have created your `.gitignore` file (see below) and pushed all necessary source files to your repository.

### 1. Repository & Environment Setup

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
````

Create a file named **`.env`** in the root directory to configure the pipeline (do NOT commit this file):

```env
# .env
TICKERS="AAPL,AMZN,GOOGL,MSFT,TSLA"
REFRESH_INTERVAL=60
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

👉 **http://127.0.0.1:8050**

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
| **Data Flow** | **Historical:** Fetches **30 days** of **30-minute** data (`*_hist.csv`) once. A **30-second delay** between tickers mitigates API rate limiting. **Live:** Polls **1-minute** data for the current day (`*_live.csv`) every **60 seconds**. |
| **Key Components** | `LiveTracker`/`DataManager` for data persistence. `AlertEngine` for rule-based checks (Trend, Price Swing, Volume). `LLMInterface` for synchronous Ollama communication. |
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
| **Visualization** | Displays two pairs of charts (Historical Price/Volume and Live Price/Volume) and real-time alerts. |
| **RAG Chatbot Logic** | Uses **`search_rag_index`** to retrieve semantic context. Sends the chat history and retrieved context to the **Ollama** LLM using a **strict system prompt** (16 rules) to ensure a professional, data-driven response. |
| **Local Limitation** | The `chat_llm` callback runs **synchronously** on the main thread, causing the UI to freeze momentarily during the LLM's response generation. |

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
