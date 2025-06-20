# Healthcare Chatbot using GPT-2

A domain-specific generative chatbot designed to answer healthcare-related questions using a fine-tuned GPT-2 model. Built with a FastAPI backend and React + Tailwind frontend.

---

## 🧠 Project Overview

This project aims to improve access to reliable healthcare information by offering an AI-powered chatbot capable of answering medical queries with contextual relevance. The model is trained on the `llama3_medquad_instruct_dataset` from Hugging Face, leveraging GPT-2 for high-quality text generation.

---

## 🔍 Dataset

- **Name**: `llama3_medquad_instruct_dataset`  
- **Source**: [Hugging Face Datasets](https://huggingface.co/datasets)  
- **Description**: Instruction-tuned healthcare Q&A pairs covering diseases, symptoms, treatments, and medications.

---

## 🏗️ Architecture

- **Model**: GPT-2 (124M parameters)  
- **Training Framework**: Hugging Face Transformers (PyTorch)

---

## 🚀 Features

- Accurate and fluent healthcare-related responses  
- Zero-shot filtering for irrelevant queries  
- FastAPI backend serving the GPT-2 model  
- Modern React/Tailwind chatbot UI with chat history and reset  
- Deployed frontend (Vercel) and backend (local or cloud-ready)

---

## 🧪 Performance Metrics

| Metric     | Score |
|------------|-------|
| Perplexity | ~4.2  |
| BLEU       | 0.61  |
| F1 Score   | 0.77  |

---

## 💬 Example Interactions

**Q:** What is the treatment for type 2 diabetes?  
**A:** Type 2 diabetes is typically managed with lifestyle changes, including diet and exercise, along with oral medications like metformin...

**Q:** What is the capital of France?  
**A:** Sorry, I can only help with healthcare-related questions.

---

## ⚙️ How to Run Locally

### Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend (React)
```bash
cd frontend
npm install
npm run dev
```

---

## 📁 Project Structure

```
├── backend/
│   └── main.py
│   └── model/
├── frontend/
│   └── components/, pages/, App.jsx
├── data/
│   └── llama3_medquad_instruct_dataset.json
├── evaluation/
│   └── metrics.py
```

---

## 📹 Demo Video

*Coming soon...*

---

## 📜 License

This project is for educational and research purposes only. It is not intended for clinical use.

---

## 👥 Contributors

- [Your Name]
- Dataset from Hugging Face
- Model from OpenAI (GPT-2)

---

## 📬 Contact

For questions or collaboration: [your_email@example.com]
