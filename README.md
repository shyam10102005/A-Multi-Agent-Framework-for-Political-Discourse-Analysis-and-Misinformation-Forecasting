# 🧠 Political Intelligence System
**A Multi-Agent Framework for Political Discourse Analysis and Misinformation Forecasting**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview
[cite_start]This framework is an integrated AI-driven solution designed to verify political claims, evaluate the rhetorical quality of debates, and forecast the spread of misinformation[cite: 780, 804]. [cite_start]By combining **Retrieval-Augmented Generation (RAG)** with multi-agent orchestration, the system provides real-time, evidence-based insights into evolving political narratives[cite: 781, 1023].

## 🚀 Key Features

### 1. 🔍 Automated Fact-Checking
* [cite_start]**Dynamic RAG Pipeline**: Aggregates real-time evidence from **Tavily Search** and **Wikipedia**[cite: 72, 73, 1022].
* [cite_start]**Intelligent Reranking**: Uses a **Cross-Encoder** (MS MARCO MiniLM) to ensure only the most relevant evidence is processed[cite: 165, 1114].
* [cite_start]**Explainable Verdicts**: Generates verdicts (True/False/Misleading/Unverified) using **Google's Flan-T5** with grounded reasoning to minimize hallucinations[cite: 136, 164, 1115].

### 2. 🗣 Debate & Rhetoric Analysis
* [cite_start]**Sentiment & Stance**: Identifies emotional polarity and partisan framing using fine-tuned **DistilBERT** models[cite: 166, 1119].
* [cite_start]**Fallacy Detection**: Heuristically detects logical fallacies such as *Slippery Slope*, *False Dilemma*, and *Appeal to Fear*[cite: 141, 1120].

### 3. 📈 Misinformation Forecaster
* [cite_start]**Trajectory Tracking**: Visualizes the historical evolution of political claims over time[cite: 12, 1134].
* [cite_start]**Risk Assessment**: Calculates a **Misinformation Spread Risk Score** based on recurrence frequency and source credibility[cite: 105, 1126].

---

## 🏗 System Architecture

The system operates on a modular pipeline:
1.  [cite_start]**Ingestion**: Normalizes raw text and identifies check-worthy claims[cite: 61, 1016].
2.  [cite_start]**Retrieval**: Multi-source gathering and DiskCache-optimized lookup[cite: 75, 127].
3.  [cite_start]**Verification**: Cross-encoder reranking followed by LLM-based reasoning[cite: 77, 82].
4.  [cite_start]**Analytics**: Parallel processing of sentiment, bias, and temporal drift[cite: 139, 147].

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
* [cite_start]**Core**: Python 3.10+, PyTorch [cite: 160]
* [cite_start]**Models**: Flan-T5 (Reasoning), DistilBERT (Sentiment), MiniLM (Reranking) 
* [cite_start]**Data**: Tavily Search API, Wikipedia API [cite: 126]
* [cite_start]**UI/UX**: Gradio [cite: 128]

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue for new fallacy detection heuristics or additional data sources.
