from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
import logging
import sys
from fastapi.middleware.cors import CORSMiddleware

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Healthcare Chatbot API",
    description="A FastAPI-based healthcare chatbot using GPT-2 fine-tuned on medical data, with zero-shot classification to filter non-healthcare queries.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    domain_confidence: str
    is_healthcare: bool

# Resolve model path
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "model"))
logger.info(f"Resolved MODEL_PATH: {MODEL_PATH}")
logger.info(f"Python sys.path: {sys.path}")
logger.info(f"Current working directory: {os.getcwd()}")

# Verify model directory exists
if not os.path.isdir(MODEL_PATH):
    raise Exception(f"Model directory does not exist: {MODEL_PATH}")

# Verify required files
required_files = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json"
]
missing_files = [f for f in required_files if not os.path.isfile(os.path.join(MODEL_PATH, f))]
if missing_files:
    raise Exception(f"Missing required files in {MODEL_PATH}: {missing_files}")

# Log file presence
for f in required_files:
    file_path = os.path.join(MODEL_PATH, f)
    logger.info(f"File {f} exists: {os.path.isfile(file_path)}, Size: {os.path.getsize(file_path) if os.path.isfile(file_path) else 'N/A'} bytes")

# Load tokenizer and model
try:
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    logger.info("Tokenizer loaded successfully")
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model or tokenizer: {str(e)}")
    raise Exception(f"Failed to load model or tokenizer: {str(e)}")

# Load zero-shot classification pipeline
try:
    logger.info("Loading zero-shot classifier...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    logger.info("Zero-shot classifier loaded successfully")
except Exception as e:
    logger.error(f"Failed to load zero-shot classifier: {str(e)}")
    raise Exception(f"Failed to load zero-shot classifier: {str(e)}")

# Domain label and threshold
DOMAIN_LABEL = "healthcare"
THRESHOLD = 0.7

def is_healthcare_query(text: str) -> tuple[bool, float]:
    """Check if the query is healthcare-related using zero-shot classification."""
    result = classifier(text, candidate_labels=[DOMAIN_LABEL])
    score = result["scores"][0]
    return score >= THRESHOLD, score

@app.post("/ask", response_model=QueryResponse, summary="Ask a healthcare question")
async def ask_question(request: QueryRequest):
    """
    Submit a question to the healthcare chatbot. The chatbot will respond only to healthcare-related questions,
    determined by a zero-shot classifier. Non-healthcare questions will be rejected.

    - **question**: The user's question (string)
    - Returns: Answer, domain confidence, and whether the question is healthcare-related
    """
    try:
        user_input = request.question.strip()
        if not user_input:
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Check if the query is healthcare-related
        is_relevant, confidence = is_healthcare_query(user_input)
        confidence_pct = f"{confidence * 100:.2f}%"

        if is_relevant:
            # Generate GPT-2 response
            prompt = f"User: {user_input}\nBot:"
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

            with torch.no_grad():
                output = model.generate(
                    inputs.input_ids,
                    max_length=100,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7
                )

            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            bot_reply = decoded.split("Bot:")[-1].strip()
            return QueryResponse(
                answer=bot_reply,
                domain_confidence=confidence_pct,
                is_healthcare=True
            )
        else:
            return QueryResponse(
                answer="Sorry, I can only answer healthcare-related questions.",
                domain_confidence=confidence_pct,
                is_healthcare=False
            )

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/", summary="Health check")
async def health_check():
    """Check if the API is running."""
    return {"status": "API is running"}