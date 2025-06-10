
# 🇻🇳 Vietnamese Legal Chatbot

A Retrieval-Augmented Generation (RAG) powered chatbot designed to assist users in querying Vietnamese law—especially on **Cybersecurity, Network Safety**, and **Information Systems**—in a friendly and legally accurate way.


## 🚀 Features

- ✅ **RAG Pipeline**: Combines query understanding, semantic vector search, and LLM-based generation.
- 🔍 **Hybrid Search**: Leverages both semantic search and keyword search to improve retrieval accuracy.
- 🤖 **LLM Integration**: Uses models from **Ollama** (e.g., `llama3.1:8b`) via LangChain.
- 📚 **Legal Corpus**: Over 100 documents from official legal sources, chunked into ~9,700 pieces.
- 🧠 **Embedding Fine-tuning**: Option to fine-tune retrieval embeddings for better domain alignment.
- 🧪 **Chatbot Answer Evaluation**: Use LLMs to assess chatbot answers for quality and relevance.
- 💬 **Messenger-ready**: Easily deploy on Facebook Messenger using FastAPI and Ngrok.

## 🧪 Run the Application

### 1️⃣ Requirements

- Python `3.11.11`
- [Ollama](https://ollama.com/) installed locally
- Vector DB backend: `ChromaDB`
- Local data in `./data/corpus/`

### 2️⃣ Clone the Repo

```bash
git clone https://github.com/cssi87m/Chatbot-for-Vietnamese-Law.git
cd Chatbot-for-Vietnamese-Law
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Pull the LLM Model

Example (using Ollama):

```bash
ollama pull llama3.1:8b
```

---

## 🔁 Prepare the Vector DB

```bash
# Generate or update vector database
python populate_database.py

# Or reset and rebuild from scratch
python populate_database.py --reset
```

---

## 💬 Ask a Legal Question (CLI)

```bash
python infer.py
# Then type your legal question in Vietnamese
```

---

## 🧠 Finetune Embedding Model (Optional)

```bash
python finetune_embedding.py \
  --train_dir train.csv \
  --eval_dir eval.csv \
  --model_path model.pt \
  --output ./checkpoints \
  --epochs 10 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --device cuda
```

---

## 🧪 Evaluate Chatbot Responses

```bash
python evaluate_chatbot.py --test_dataset_dir ./test_data.csv
```

---

## 🌐 Run as FastAPI App

```bash
uvicorn app:app --reload
```

Then expose your app to Facebook using:

```bash
ngrok http 8000
```

Access API docs via: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📄 Environment Variables

Put these in `.env` file:

```env
FB_ACCESS_TOKEN=your_page_access_token
FB_VERIFY_TOKEN=your_verify_token
FB_APP_SECRET=your_app_secret
```

---

## 📬 Messenger Integration

- `/webhook/ [GET]`: Verify Facebook webhook.
- `/webhook/ [POST]`: Handle incoming messages from Messenger.
- Uses `X-Hub-Signature` to validate Facebook's authenticity.


