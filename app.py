# ==============================================================================
#  SentimentAI — app.py
#  Production-ready Flask backend for Sentiment Analysis (6 emotions)
#  Author : Your Name
#  Version: 1.0.0
#  License: MIT
# ==============================================================================

# ── SECTION 1: Imports & Configuration ──────────────────────────────────────── #imports
import os
import re
import sys
import pickle
import logging
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ── Optional: NLTK stopwords (comment out if not installed) ─────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    # Download stopwords on first run (comment out after first run)
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
    USE_STOPWORDS = True
except ImportError:
    STOP_WORDS = set()
    USE_STOPWORDS = False
    print("[WARNING] NLTK not installed. Stopword removal disabled.")

# ==============================================================================
# App Initialization & Configuration
# ==============================================================================
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (allows frontend on different port)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),               # Console output
        logging.FileHandler("app.log", encoding="utf-8") # Persistent log file
    ]
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VEC_PATH   = os.path.join(BASE_DIR, "vectorizer.pkl")
MAX_CHARS  = 1000   # Maximum allowed input length
MIN_CHARS  = 3      # Minimum allowed input length

# ==============================================================================
# SECTION 2: Emotion Label Mapping                                     #labels
# ==============================================================================
#
#  Maps the integer output of your trained model to a human-readable
#  emotion label, emoji, hex colour, and a short description.
#  ┌──────┬───────────┬───────┬──────────┐
#  │  ID  │  Emotion  │ Emoji │  Colour  │
#  ├──────┼───────────┼───────┼──────────┤
#  │   0  │  Sadness  │  😢   │ #6EA8FE  │
#  │   1  │  Joy      │  😊   │ #20C997  │
#  │   2  │  Love     │  ❤️   │ #F06292  │
#  │   3  │  Anger    │  😡   │ #FF6B6B  │
#  │   4  │  Fear     │  😨   │ #A371F7  │
#  │   5  │  Surprise │  😲   │ #FFA64D  │
#  └──────┴───────────┴───────┴──────────┘
#
EMOTION_LABELS = {
    0: {
        "label"      : "Sadness",
        "emoji"      : "😢",
        "color"      : "#6EA8FE",
        "description": "The text expresses feelings of sorrow or unhappiness.",
        "intensity"  : "low"
    },
    1: {
        "label"      : "Joy",
        "emoji"      : "😊",
        "color"      : "#20C997",
        "description": "The text radiates happiness, delight, or excitement.",
        "intensity"  : "high"
    },
    2: {
        "label"      : "Love",
        "emoji"      : "❤️",
        "color"      : "#F06292",
        "description": "The text conveys affection, warmth, or romantic feeling.",
        "intensity"  : "medium"
    },
    3: {
        "label"      : "Anger",
        "emoji"      : "😡",
        "color"      : "#FF6B6B",
        "description": "The text shows frustration, rage, or strong displeasure.",
        "intensity"  : "high"
    },
    4: {
        "label"      : "Fear",
        "emoji"      : "😨",
        "color"      : "#A371F7",
        "description": "The text reflects anxiety, dread, or apprehension.",
        "intensity"  : "medium"
    },
    5: {
        "label"      : "Surprise",
        "emoji"      : "😲",
        "color"      : "#FFA64D",
        "description": "The text expresses shock, astonishment, or unexpectedness.",
        "intensity"  : "medium"
    }
}

# ==============================================================================
# SECTION 3: Model & Vectorizer Loading                                 #model
# ==============================================================================
model      = None   # Scikit-learn classifier (e.g. LogisticRegression / SVM)
vectorizer = None   # TF-IDF vectorizer
model_loaded = False


def load_artifacts() -> bool:
    """
    Load model.pkl and vectorizer.pkl from disk into global variables.

    Returns:
        bool: True if both artifacts loaded successfully, False otherwise.

    Raises:
        FileNotFoundError : If .pkl files are missing.
        pickle.UnpicklingError: If files are corrupted.
    """
    global model, vectorizer, model_loaded

    logger.info("=" * 60)
    logger.info("Loading ML artifacts …")

    # ── Load Model ──────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        logger.error(f"model.pkl not found at: {MODEL_PATH}")
        return False

    if not os.path.exists(VEC_PATH):
        logger.error(f"vectorizer.pkl not found at: {VEC_PATH}")
        return False

    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logger.info(f"✅  Model loaded        → {type(model).__name__}")

        with open(VEC_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        logger.info(f"✅  Vectorizer loaded   → {type(vectorizer).__name__}")

        model_loaded = True
        logger.info("=" * 60)
        return True

    except pickle.UnpicklingError as e:
        logger.error(f"Corrupted pickle file: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error loading artifacts: {e}")
        return False


# ==============================================================================
# SECTION 4: Text Preprocessing Pipeline                             #preprocess
# ==============================================================================

def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline applied before TF-IDF transformation.

    Pipeline steps (in order):
        1. Strip leading/trailing whitespace
        2. Convert to lowercase
        3. Remove URLs (http/https/www)
        4. Remove HTML tags (, 
, etc.)
        5. Remove email addresses
        6. Remove punctuation and special characters
        7. Remove digits / numbers
        8. Collapse multiple whitespace into single space
        9. (Optional) Remove NLTK stopwords

    Args:
        text (str): Raw input string from the user.

    Returns:
        str: Cleaned, normalised text ready for TF-IDF vectorisation.

    Example:
        >>> preprocess_text("I LOVE you so much!!! 😍 Visit http://example.com")
        'love much visit'          # (with stopwords enabled)
        'i love you so much visit' # (without stopwords)
    """
    if not isinstance(text, str):
        text = str(text)

    # Step 1 — Strip whitespace
    text = text.strip()

    # Step 2 — Lowercase
    text = text.lower()

    # Step 3 — Remove URLs (http, https, www)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Step 4 — Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Step 5 — Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # Step 6 — Remove punctuation & special characters (keep letters + spaces)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Step 7 — Remove standalone digits
    text = re.sub(r"\b\d+\b", " ", text)

    # Step 8 — Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()

    # Step 9 — Remove stopwords (optional, based on NLTK availability)
    if USE_STOPWORDS and STOP_WORDS:
        words = text.split()
        words = [w for w in words if w not in STOP_WORDS and len(w) > 1]
        text  = " ".join(words)

    return text


def validate_input(text: str) -> tuple[bool, str]:
    """
    Validate user input before processing.

    Args:
        text (str): Raw input from the request.

    Returns:
        tuple[bool, str]: (is_valid, error_message)
                          error_message is empty string if valid.
    """
    if not text or not text.strip():
        return False, "Input text cannot be empty."

    if len(text.strip()) < MIN_CHARS:
        return False, f"Input must be at least {MIN_CHARS} characters long."

    if len(text) > MAX_CHARS:
        return False, f"Input must not exceed {MAX_CHARS} characters."

    return True, ""


# ==============================================================================
# SECTION 5: Flask Routes                                               #routes
# ==============================================================================

@app.route("/")                           # ── Home Page
def index():
    """
    Serve the main HTML page (templates/index.html).

    Returns:
        Rendered HTML template.
    """
    return render_template("index.html")


# ── ──────────────────────────────────────────────────────────────────────── ──
# SECTION 6: /predict Endpoint (Core Functionality)                   #predict
# ── ──────────────────────────────────────────────────────────────────────── ──

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    ─────────────
    Accept a JSON body with a "text" field, run the ML pipeline, and return
    the predicted emotion with confidence scores.

    Request Body (JSON):
        {
            "text": "I am feeling so happy today!"
        }

    Success Response (200):
        {
            "success"       : true,
            "input_text"    : "I am feeling so happy today!",
            "cleaned_text"  : "feeling happy today",
            "prediction"    : {
                "id"         : 1,
                "label"      : "Joy",
                "emoji"      : "😊",
                "color"      : "#20C997",
                "description": "The text radiates happiness, delight, or excitement.",
                "confidence" : 94.7
            },
            "all_scores"    : [
                {"id": 0, "label": "Sadness",  "emoji": "😢", "score": 1.2},
                {"id": 1, "label": "Joy",      "emoji": "😊", "score": 94.7},
                ...
            ],
            "timestamp"     : "2024-01-15 14:32:07"
        }

    Error Responses:
        400 — Validation error (empty / too long input)
        415 — Wrong Content-Type (must be application/json)
        503 — Model not loaded
        500 — Internal server error
    """
    # ── Guard: Model must be loaded ─────────────────────────────────────────
    if not model_loaded:
        logger.error("Prediction attempted but model is not loaded.")
        return jsonify({
            "success": False,
            "error"  : "ML model is not available. Please check server logs.",
            "code"   : 503
        }), 503

    # ── Guard: Content-Type must be JSON ────────────────────────────────────
    if not request.is_json:
        return jsonify({
            "success": False,
            "error"  : "Content-Type must be 'application/json'.",
            "code"   : 415
        }), 415

    # ── Parse request body ──────────────────────────────────────────────────
    data        = request.get_json(silent=True) or {}
    raw_text    = data.get("text", "")

    # ── Input validation ────────────────────────────────────────────────────
    is_valid, error_msg = validate_input(raw_text)
    if not is_valid:
        logger.warning(f"Invalid input: {error_msg!r}")
        return jsonify({
            "success": False,
            "error"  : error_msg,
            "code"   : 400
        }), 400

    try:
        # ── Step A: Preprocess ───────────────────────────────────────────────
        cleaned_text = preprocess_text(raw_text)
        logger.info(f"Raw    : {raw_text[:80]!r}")
        logger.info(f"Cleaned: {cleaned_text[:80]!r}")

        # ── Step B: TF-IDF Vectorisation ─────────────────────────────────────
        #    transform() — NOT fit_transform() — to use training vocabulary
        tfidf_matrix = vectorizer.transform([cleaned_text])

        # ── Step C: Prediction ───────────────────────────────────────────────
        predicted_class = int(model.predict(tfidf_matrix)[0])

        # ── Step D: Confidence Scores ─────────────────────────────────────────
        #    Use predict_proba() if the model supports it (most Scikit-learn
        #    classifiers do when probability=True); fall back to decision_function
        #    + softmax otherwise.
        if hasattr(model, "predict_proba"):
            raw_probs   = model.predict_proba(tfidf_matrix)[0]  # shape: (n_classes,)
            probs       = (raw_probs * 100).tolist()             # convert to %

        elif hasattr(model, "decision_function"):
            # Softmax fallback for SVMs / linear models
            scores      = model.decision_function(tfidf_matrix)[0]
            scores      = np.array(scores, dtype=float)
            exp_scores  = np.exp(scores - scores.max())          # numerical stability
            softmax     = exp_scores / exp_scores.sum()
            probs       = (softmax * 100).tolist()

        else:
            # Last-resort: 100% to predicted class, 0% to rest
            n_classes   = len(EMOTION_LABELS)
            probs       = [0.0] * n_classes
            probs[predicted_class] = 100.0

        # ── Step E: Build per-class score list ───────────────────────────────
        all_scores = []
        for class_id, meta in EMOTION_LABELS.items():
            score_val = round(probs[class_id], 2) if class_id < len(probs) else 0.0
            all_scores.append({
                "id"   : class_id,
                "label": meta["label"],
                "emoji": meta["emoji"],
                "color": meta["color"],
                "score": score_val
            })

        # Sort descending by score for display
        all_scores.sort(key=lambda x: x["score"], reverse=True)

        # ── Step F: Compose response ──────────────────────────────────────────
        emotion_meta     = EMOTION_LABELS[predicted_class]
        confidence_score = round(probs[predicted_class], 2)

        response = {
            "success"    : True,
            "input_text" : raw_text,
            "cleaned_text": cleaned_text,
            "prediction" : {
                "id"         : predicted_class,
                "label"      : emotion_meta["label"],
                "emoji"      : emotion_meta["emoji"],
                "color"      : emotion_meta["color"],
                "description": emotion_meta["description"],
                "confidence" : confidence_score
            },
            "all_scores" : all_scores,
            "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        logger.info(
            f"Prediction: {emotion_meta['label']} "
            f"({confidence_score:.1f}%) ← {raw_text[:50]!r}"
        )
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error"  : "An internal error occurred during prediction.",
            "code"   : 500
        }), 500


# ── ──────────────────────────────────────────────────────────────────────── ──
# SECTION 7: Health-Check Endpoint                                     #health
# ── ──────────────────────────────────────────────────────────────────────── ──

@app.route("/health", methods=["GET"])
def health():
    """
    GET /health
    ───────────
    Lightweight liveness probe used by Render, Railway, and Docker.

    Response (200 — healthy):
        {
            "status"      : "healthy",
            "model_loaded": true,
            "timestamp"   : "2024-01-15 14:32:07",
            "version"     : "1.0.0"
        }

    Response (503 — model not loaded):
        {
            "status"      : "degraded",
            "model_loaded": false,
            ...
        }
    """
    status_code = 200 if model_loaded else 503
    return jsonify({
        "status"      : "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "timestamp"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version"     : "1.0.0"
    }), status_code


@app.route("/api/labels", methods=["GET"])
def get_labels():
    """
    GET /api/labels
    ───────────────
    Return all emotion labels and metadata. Useful for dynamic frontends.

    Response (200):
        {
            "labels": [
                {"id": 0, "label": "Sadness", "emoji": "😢", "color": "#6EA8FE"},
                ...
            ]
        }
    """
    labels = [
        {
            "id"   : k,
            "label": v["label"],
            "emoji": v["emoji"],
            "color": v["color"],
            "description": v["description"]
        }
        for k, v in EMOTION_LABELS.items()
    ]
    return jsonify({"labels": labels}), 200


# ── ──────────────────────────────────────────────────────────────────────── ──
# SECTION 8: Error Handlers
# ── ──────────────────────────────────────────────────────────────────────── ──

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Route not found.", "code": 404}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"success": False, "error": "HTTP method not allowed.", "code": 405}), 405

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Unhandled 500 error: {e}")
    return jsonify({"success": False, "error": "Internal server error.", "code": 500}), 500


# ==============================================================================
# SECTION 9: Application Entry Point                                     #main
# ==============================================================================

if __name__ == "__main__":
    """
    Entry point when running directly with:  python app.py

    For production, use Gunicorn instead:
        gunicorn app:app --bind 0.0.0.0:5000 --workers 2 --timeout 120

    Environment variables (optional):
        PORT        — Port to listen on          (default: 5000)
        FLASK_DEBUG — Enable debug mode          (default: False)
        HOST        — Host to bind to            (default: 0.0.0.0)
    """
    # ── Load ML artifacts before accepting requests ──────────────────────────
    success = load_artifacts()
    if not success:
        logger.warning(
            "⚠️  Server starting WITHOUT a loaded model. "
            "Prediction endpoints will return 503 until model is available."
        )

    # ── Read config from environment (Railway / Render inject PORT) ──────────
    port       = int(os.environ.get("PORT", 5000))
    host       = os.environ.get("HOST", "0.0.0.0")
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    logger.info(f"🚀  Starting SentimentAI Flask server …")
    logger.info(f"    Host  : {host}")
    logger.info(f"    Port  : {port}")
    logger.info(f"    Debug : {debug_mode}")
    logger.info(f"    URL   : http://localhost:{port}")

    app.run(
        host  = host,
        port  = port,
        debug = debug_mode
    )

# ==============================================================================
# END OF FILE — app.py
# ==============================================================================