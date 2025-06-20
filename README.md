# chatbot

Healthcare Chatbot using GPT-2

A domain-specific generative chatbot designed to answer healthcare-related questions using a fine-tuned GPT-2 model. Built with a FastAPI backend and React + Tailwind frontend.

 Project Overview

This project aims to improve access to reliable healthcare information by offering an AI-powered chatbot capable of answering medical queries with contextual relevance. The model is trained on the llama3_medquad_instruct_dataset from Hugging Face, leveraging GPT-2 for high-quality text generation.

 Dataset

Name: llama3_medquad_instruct_dataset

Source: Hugging Face Datasets

Description: Instruction-tuned healthcare Q&A pairs covering diseases, symptoms, treatments, and medications.

 Architecture

Model: GPT-2 (124M parameters)

Training Framework: Hugging Face Transformers (PyTorch)

Evaluation Metrics: Perplexity, BLEU, F1-score, and qualitative testing

Out-of-domain Detection: BART-based Zero-shot Classifier (facebook/bart-large-mnli)
