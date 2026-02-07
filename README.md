# 🧠 A Multi-Agent Framework for Political Discourse Analysis and Misinformation Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview
This framework is an integrated AI-driven solution designed to verify political claims, evaluate the rhetorical quality of debates, and forecast the spread of misinformation. By combining **Retrieval-Augmented Generation (RAG)** with multi-agent orchestration, the system provides real-time, evidence-based insights into evolving political narratives.

## 🚀 Key Features

### 1. 🔍 Automated Fact-Checking
* **Dynamic RAG Pipeline**: Aggregates real-time evidence from **Tavily Search** and **Wikipedia**.
* **Intelligent Reranking**: Uses a **Cross-Encoder** (MS MARCO MiniLM) to ensure only the most relevant evidence is processed.
* **Explainable Verdicts**: Generates verdicts (True/False/Misleading/Unverified) using **Google's Flan-T5** with grounded reasoning to minimize hallucinations.

### 2. 🗣 Debate & Rhetoric Analysis
* **Sentiment & Stance**: Identifies emotional polarity and partisan framing using fine-tuned **DistilBERT** models.
* **Fallacy Detection**: Heuristically detects logical fallacies such as *Slippery Slope*, *False Dilemma*, and *Appeal to Fear*.

### 3. 📈 Misinformation Forecaster
* **Trajectory Tracking**: Visualizes the historical evolution of political claims over time.
* **Risk Assessment**: Calculates a **Misinformation Spread Risk Score** based on recurrence frequency and source credibility.

---

## 🏗 System Architecture

The system operates on a modular pipeline:
1.  **Ingestion**: Normalizes raw text and identifies check-worthy claims.
2.  **Retrieval**: Multi-source gathering and DiskCache-optimized lookup.
3.  **Verification**: Cross-encoder reranking followed by LLM-based reasoning.
4.  **Analytics**: Parallel processing of sentiment, bias, and temporal drift.

---

## 💻 Installation & Setup

### Local Setup
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/shyam10102005/A-Multi-Agent-Framework-for-Political-Discourse-Analysis-and-Misinformation-Forecasting.git
    cd A-Multi-Agent-Framework-for-Political-Discourse-Analysis-and-Misinformation-Forecasting
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure Environment**:
    Add your **Tavily API Key** to your environment or `main.py`:
    ```python
    os.environ["TAVILY_API_KEY"] = "your_api_key_here"
    ```
4.  **Run Application**:
    ```bash
    python main.py
    ```

### Google Colab Setup (Recommended for GPU Access)
1.  Open a new notebook and set the runtime to **T4 GPU**.
2.  Install requirements:
    ```python
    !pip install gradio transformers sentence-transformers tavily-python wikipedia diskcache torch --quiet
    ```
3.  Load your key securely:
    ```python
    from google.colab import userdata
    os.environ["TAVILY_API_KEY"] = userdata.get('TAVILY_API_KEY')
    ```

---

## 📚 Tech Stack
* **Core**: Python 3.10+, PyTorch
* **Models**: Flan-T5 (Reasoning), DistilBERT (Sentiment), MiniLM (Reranking) 
* **Data**: Tavily Search API, Wikipedia API
* **UI/UX**: Gradio

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue for new fallacy detection heuristics or additional data sources.
