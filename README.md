
# 🎌 AI Anime Recommendation System

An end-to-end **AI-powered Anime Recommendation System** built using **LangChain**, **Groq LLMs**, and **ChromaDB** with an interactive **Streamlit** interface.
This project showcases how to build and deploy a **production-ready LLM application** featuring vector search, prompt engineering, and Kubernetes-based deployment on **GCP**.

---

## 🚀 Key Features

- 🔎 **Semantic Anime Recommendations** using vector embeddings
- 🧠 **LLM-powered reasoning** with Groq (low-latency inference)
- 📦 **Vector Store** powered by ChromaDB
- 🖥️ **Interactive UI** built with Streamlit
- 🧩 Modular, scalable project structure
- 🐳 Dockerized application
- ☸️ Kubernetes deployment (Minikube / GKE-ready)
- 📊 Monitoring with Grafana Cloud
- 🔐 Secure configuration via environment variables

---

## 🏗️ Tech Stack

- **Language:** Python 3.10+
- **LLM Framework:** LangChain
- **LLM Provider:** Groq
- **Embeddings:** HuggingFace
- **Vector Database:** ChromaDB
- **Frontend:** Streamlit
- **Containerization:** Docker
- **Orchestration:** Kubernetes
- **Cloud:** Google Cloud Platform (GCP)
- **Monitoring:** Grafana Cloud

---

## 📁 Project Structure

```bash
.
├── app/
│   ├── app.py              # Streamlit entry point
│   └── __init__.py
├── chroma_db/              # Persistent vector store
├── config/                 # App & model configurations
├── data/                   # Anime datasets
├── logs/                   # Application logs
├── pipeline/
│   ├── build_pipeline.py
│   └── pipeline.py
├── src/
│   ├── data_loader.py      # Data ingestion & preprocessing
│   ├── vector_store.py     # ChromaDB integration
│   ├── prompt_template.py  # Prompt engineering
│   ├── recommender.py      # Recommendation logic
│   └── __init__.py
├── utils/
│   ├── logger.py
│   └── custom_exception.py
├── .env
├── Dockerfile
├── imlops-k8s.yaml
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Sumit-Prasad01/ai-anime-recommender.git
cd ai-anime-recommender
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables
Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key
HUGGINGFACE_API_KEY=your_hf_api_key
```

---

## ▶️ Running the Application (Streamlit)

```bash
streamlit run app/app.py
```

Then open:
```
http://localhost:8501
```

---

## 🧠 Recommendation Pipeline

1. Load & preprocess anime dataset
2. Generate embeddings using HuggingFace models
3. Store embeddings in ChromaDB
4. Accept user preferences via Streamlit UI
5. Perform semantic similarity search
6. Use Groq LLM to generate explainable recommendations

---

## 🐳 Docker Usage

```bash
docker build -t anime-recommender .
docker run -p 8501:8501 --env-file .env anime-recommender
```

---

## ☸️ Kubernetes Deployment

```bash
kubectl apply -f imlops-k8s.yaml
```

Verify:
```bash
kubectl get pods
kubectl get svc
```

---

## 📊 Monitoring

- Integrated with **Grafana Cloud**
- Monitors:
  - Application latency
  - Resource utilization
  - Error rates
  - Container health

---

## 🧪 Future Enhancements

- User-based personalization
- Feedback-aware recommendations
- RAG with anime reviews & summaries
- Streaming & session-based memory
- GKE Autopilot deployment

---
