# 🤖 Universal Omni-Agent (Research & Multi-Task Chatbot)

A powerful, multi-agent AI system built with **FastAPI**, **Streamlit**, and **LangGraph**. This intelligent assistant routes your queries dynamically to specialized sub-agents, capable of performing real-time web searches, querying financial data, fetching weather, retrieving academic papers from ArXiv, and analyzing uploaded PDF documents using Retrieval-Augmented Generation (RAG).

![Omni-Agent UI Preview](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit)
![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi)
![AI Framework](https://img.shields.io/badge/AI-LangGraph-purple?style=flat-square)

---

## ✨ Features

- **🧠 Intelligent Routing**: Automatically routes questions to the appropriate tool (General Chat, RAG, Web Search, or Finance) using a zero-shot Llama 3.1-8b router.
- **📚 RAG & Document Analysis**: Upload PDFs and instantly chat with your documents. Powered by **Pinecone** for lightning-fast semantic vector search and **FastEmbed** for efficient embeddings without heavy dependencies like PyTorch.
- **📄 ArXiv Research Integration**: Ask about academic topics and instantly retrieve detailed abstracts and PDF links directly from ArXiv.
- **🌐 Real-Time Web Search**: Uses DuckDuckGo to search the live internet for recent news and general queries.
- **📈 Finance & Live Sports**: Get up-to-date stock prices via Yahoo Finance and live cricket/sports scores.
- **☀️ Live Weather Forecasts**: Integrated with Open-Meteo for accurate, current weather data worldwide.
- **⚡ Fast SSE Streaming**: Responses are streamed word-by-word to the UI for a seamless, ChatGPT-like experience.
- **🎨 Modern Dark UI**: A beautiful, custom-styled Streamlit interface featuring history tracking, session management, and visual route indicators.

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend API**: FastAPI (Python)
- **AI Models**: Groq (Llama 3.3-70b for reasoning, Llama 3.1-8b for routing)
- **Agent Orchestration**: LangChain & LangGraph
- **Vector Database**: Pinecone (Serverless Cloud Vector DB)
- **Embeddings**: FastEmbed (ONNX, lightweight, fast)
- **Persistent Storage**: SQLite (SQLAlchemy & aiosqlite)

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- A [Groq API Key](https://console.groq.com/keys)
- A [Pinecone API Key](https://www.pinecone.io/)

### 2. Installation

Clone the repository:
```bash
git clone https://github.com/Abhinav-1612/Mutlitask_chatbot.git
cd Mutlitask_chatbot
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory and add your API keys:
```env
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### 4. Running the Application
You will need to run both the FastAPI backend and the Streamlit frontend.

**Start the FastAPI Backend:**
```bash
uvicorn app.main:app --reload
```
The backend will run on `http://localhost:8000`.

**Start the Streamlit Frontend:**
Open a new terminal window and run:
```bash
streamlit run streamlit_app.py
```
The UI will be accessible at `http://localhost:8501`.

## 📁 Project Structure

```text
.
├── app/
│   ├── agents/          # LangGraph nodes, routing logic, and state
│   ├── api/             # FastAPI endpoints (chat, upload, ui)
│   ├── database/        # SQLite setup and Pinecone vector DB integration
│   ├── models/          # Pydantic schemas
│   ├── tools/           # External APIs (Web search, ArXiv, Weather, Finance)
│   ├── config.py        # Environment variables & configuration
│   ├── graph.py         # LangGraph workflow definition
│   ├── main.py          # FastAPI application entry point
│   └── sse.py           # Server-Sent Events parser
├── streamlit_app.py     # Main Streamlit UI frontend
├── requirements.txt     # Python dependencies
└── .env                 # Environment variables (not tracked by git)
```

## 🤖 Agent Workflow
1. **Gateway Router**: Fast, zero-shot intent classification determining if a query needs Web, Finance, RAG, or just General Chat.
2. **Supervisor**: Secondary validation net to extract tickers or refine routing before execution.
3. **Specialist Nodes**:
   - `web_node`: Scrapes duckduckgo and Open-Meteo.
   - `finance_node`: Retrieves stock data (via direct chart API) and live cricket scores.
   - `rag_node`: Performs Pinecone similarity searches and ArXiv retrievals.
   - `general_node`: Direct conversational Llama 3.3-70b interacting with basic tools.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

