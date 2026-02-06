# 🧠 Political Intelligence System

A multi-agent framework designed for political discourse analysis, automated fact-checking, and misinformation forecasting. This system leverages Large Language Models (LLMs), search APIs, and sentiment analysis to provide real-time insights into political claims and debates.

## 🚀 Features

### 1. 🔍 Automated Fact-Checking
- **Multi-Source Evidence Retrieval**: Aggregates information from **Tavily** (web search) and **Wikipedia**.
- **Rule-Based Verification**: Instantly flags inconsistencies in dates, numbers, and historical facts.
- **LLM Reasoning**: Uses **Google's Flan-T5** model to synthesize evidence and provide a reasoned verdict (User-friendly explanation).
- **Source Transparency**: Displays all sources used for verification with summaries.

### 2. 🗣 Debate Analysis
- **Sentiment Analysis**: Determines the emotional tone of political speeches or debates.
- **Bias Detection**: Identifies positive or negative bias.
- **Logical Fallacy Detection**: Detects common rhetorical fallacies like "Slippery Slope", "False Dilemma", and "Appeal to Fear".

### 3. 📈 Historical Claim Trajectory
- **Timeline Tracking**: Visualizes how a claim has evolved over time.
- **Source Frequency Analysis**: Identifies which domains are propagating the claim.
- **Misinformation Risk Assessment**: Calculates a risk level (LOW/MEDIUM/HIGH) based on the spread and nature of sources.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/political-intelligence-system.git
   cd political-intelligence-system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🔑 Configuration

You need a **Tavily API Key** for the search functionality to work.

1. Open `main.py`.
2. Find the line:
   ```python
   os.environ["TAVILY_API_KEY"] = " "
   ```
3. Insert your API key inside the quotes.

## ▶️ Usage

Run the application:
```bash
python main.py
```

The application will launch a **Gradio** interface in your browser (usually at `http://127.0.0.1:7860`).

## 📚 Tech Stack

- **Frontend**: Gradio
- **LLM**: Google Flan-T5 (`google/flan-t5-base`)
- **Search**: Tavily API, Wikipedia API
- **NLP**: Transformers (Hugging Face), Sentence Transformers
- **Sentiment Analysis**: DistilBERT
- **Caching**: DiskCache

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.
