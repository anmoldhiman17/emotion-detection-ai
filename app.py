# ============================================================
#  SentimentAI — Gradio App
#  Single-file, production-ready sentiment analysis
#  Author  : Your Name
#  Version : 2.0.0
# ============================================================

# ── Section 1: Imports ──────────────────────────────────────
import os
import re
import pickle
import logging
import numpy  as np
import pandas as pd
import gradio as gr

# ── Section 2: Logging Setup ────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Section 3: Emotion Label Map ────────────────────────────
EMOTION_MAP = {
    0: {"label": "Sadness",  "emoji": "\U0001f622", "color": "#60a5fa"},
    1: {"label": "Joy",      "emoji": "\U0001f60a", "color": "#fbbf24"},
    2: {"label": "Love",     "emoji": "\u2764\ufe0f",  "color": "#f472b6"},
    3: {"label": "Anger",    "emoji": "\U0001f621", "color": "#ef4444"},
    4: {"label": "Fear",     "emoji": "\U0001f628", "color": "#a78bfa"},
    5: {"label": "Surprise", "emoji": "\U0001f632", "color": "#34d399"},
}

# ── Section 4: Load Model & Vectorizer ──────────────────────
MODEL_PATH      = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

def load_artifacts():
    """Load model and vectorizer from disk with error handling."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            "Make sure it is in the same directory as app.py."
        )
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Vectorizer file '{VECTORIZER_PATH}' not found. "
            "Make sure it is in the same directory as app.py."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    logger.info("Model and vectorizer loaded successfully.")
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
    MODEL_LOADED = True
except FileNotFoundError as e:
    logger.error(str(e))
    model, vectorizer = None, None
    MODEL_LOADED = False

# ── Section 5: Prediction History (in-memory) ───────────────
prediction_history = []   # max 3 entries

# ── Section 6: Text Preprocessing ──────────────────────────
def preprocess(text: str) -> str:
    """
    Clean raw input text before vectorization.
    Steps:
        1. Lowercase
        2. Remove URLs
        3. Remove HTML tags
        4. Remove email addresses
        5. Remove punctuation
        6. Remove digits
        7. Collapse extra whitespace
    """
    text = text.lower()                                # 1. lowercase
    text = re.sub(r"http\S+|www\.\S+", "", text)    # 2. URLs
    text = re.sub(r"<[^>]+>", "", text)               # 3. HTML tags
    text = re.sub(r"\S+@\S+", "", text)              # 4. emails
    text = re.sub(r"[^\w\s]", " ", text)             # 5. punctuation
    text = re.sub(r"\d+", "", text)                   # 6. digits
    text = re.sub(r"\s+", " ", text).strip()          # 7. whitespace
    return text

# ── Section 7: Confidence Scores ────────────────────────────
def get_confidence_scores(model, X_vec) -> dict:
    """
    Return per-class probability scores.
    Falls back to a one-hot dict if model lacks predict_proba.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_vec)[0]
        classes = model.classes_
        scores = {int(cls): float(prob) for cls, prob in zip(classes, probs)}
    else:
        # Fallback: decision function → softmax
        if hasattr(model, "decision_function"):
            df = model.decision_function(X_vec)[0]
            e  = np.exp(df - np.max(df))
            probs = e / e.sum()
            classes = model.classes_
        else:
            pred = int(model.predict(X_vec)[0])
            classes = list(range(len(EMOTION_MAP)))
            probs   = [1.0 if i == pred else 0.0 for i in classes]
        scores = {int(cls): float(p) for cls, p in zip(classes, probs)}
    return scores

# ── Section 8: HTML Result Card Builder ─────────────────────
def build_result_card(emotion_id: int, confidence: float, scores: dict) -> str:
    """
    Build an HTML card showing the predicted emotion, confidence,
    and a mini bar chart for all 6 emotions.
    """
    info  = EMOTION_MAP[emotion_id]
    label = info["label"]
    emoji = info["emoji"]
    color = info["color"]
    pct   = round(confidence * 100, 1)

    # All-emotion bars
    bars_html = ""
    for eid in range(len(EMOTION_MAP)):
        e_info  = EMOTION_MAP[eid]
        e_score = round(scores.get(eid, 0.0) * 100, 1)
        active  = "font-weight:700;" if eid == emotion_id else ""
        bars_html += f"""
        <div style="margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;
                      font-size:13px;color:#cbd5e1;{active}">
            <span>{e_info['emoji']} {e_info['label']}</span>
            <span>{e_score}%</span>
          </div>
          <div style="background:#1e293b;border-radius:6px;height:8px;margin-top:4px;overflow:hidden;">
            <div style="width:{e_score}%;height:100%;
                        background:{e_info['color']};border-radius:6px;
                        transition:width 0.8s ease;"></div>
          </div>
        </div>"""

    card = f"""
    <div style="
        font-family:'Segoe UI',system-ui,sans-serif;
        background:linear-gradient(135deg,#0f172a,#1e1b4b);
        border:1px solid {color}44;
        border-radius:18px;padding:28px 32px;
        box-shadow:0 0 40px {color}22;
        animation:fadeIn .4s ease;">

      <style>
        @keyframes fadeIn{{from{{opacity:0;transform:translateY(8px)}}
                            to{{opacity:1;transform:translateY(0)}}}}
      </style>

      <!-- Main result -->
      <div style="text-align:center;margin-bottom:24px;">
        <div style="font-size:64px;line-height:1;margin-bottom:12px;">{emoji}</div>
        <div style="font-size:28px;font-weight:800;color:{color};
                    letter-spacing:-0.5px;">{label}</div>
        <div style="font-size:14px;color:#94a3b8;margin-top:4px;">
          Confidence: <strong style="color:#f1f5f9;">{pct}%</strong>
        </div>
      </div>

      <!-- Confidence ring bar -->
      <div style="background:#1e293b;border-radius:50px;height:10px;
                  margin-bottom:28px;overflow:hidden;">
        <div style="width:{pct}%;height:100%;
                    background:linear-gradient(90deg,{color},{color}88);
                    border-radius:50px;transition:width 1s ease;"></div>
      </div>

      <!-- All emotions -->
      <div style="border-top:1px solid #334155;padding-top:20px;">
        <div style="font-size:12px;font-weight:700;color:#64748b;
                    letter-spacing:1px;text-transform:uppercase;margin-bottom:14px;">
          All Emotion Scores
        </div>
        {bars_html}
      </div>
    </div>"""
    return card

# ── Section 9: History HTML Builder ─────────────────────────
def build_history_html() -> str:
    """Render the last 3 predictions as an HTML list."""
    if not prediction_history:
        return "<p style='color:#64748b;font-size:13px;'>No predictions yet.</p>"

    items = ""
    for i, entry in enumerate(reversed(prediction_history[-3:])):
        info  = EMOTION_MAP[entry["id"]]
        items += f"""
        <div style="
            display:flex;align-items:center;gap:14px;
            background:#0f172a;border:1px solid #1e293b;
            border-radius:12px;padding:12px 16px;margin-bottom:8px;">
          <span style="font-size:28px;">{info['emoji']}</span>
          <div style="flex:1;min-width:0;">
            <div style="font-size:13px;font-weight:700;color:{info['color']};">
              {info['label']} — {entry['confidence']}%
            </div>
            <div style="font-size:11px;color:#64748b;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
              {entry['text'][:60]}{'...' if len(entry['text'])>60 else ''}
            </div>
          </div>
          <div style="font-size:11px;color:#475569;flex-shrink:0;">#{len(prediction_history)-i}</div>
        </div>"""

    return f"""
    <div style="font-family:'Segoe UI',system-ui,sans-serif;">
      <div style="font-size:12px;font-weight:700;color:#64748b;
                  letter-spacing:1px;text-transform:uppercase;margin-bottom:12px;">
        Recent Predictions
      </div>
      {items}
    </div>"""

# ── Section 10: Main Predict Function ───────────────────────
def predict(text: str):
    """
    Full prediction pipeline:
        1. Validate input
        2. Preprocess text
        3. Vectorize
        4. Predict label + confidence
        5. Build result card HTML
        6. Update history
        7. Return all outputs
    """
    # ── Guard: model loaded? ──
    if not MODEL_LOADED:
        err = (
            "⚠️ Model files not found!\n"
            "Please place model.pkl and vectorizer.pkl "
            "in the same directory as app.py."
        )
        return err, "", build_history_html()

    # ── Guard: empty input ──
    if not text or not text.strip():
        return "⚠️ Please enter some text to analyze.", "", build_history_html()

    try:
        # 1. Preprocess
        cleaned = preprocess(text)
        if not cleaned:
            return "⚠️ Text is empty after cleaning. Try different input.", "", build_history_html()

        # 2. Vectorize
        X_vec = vectorizer.transform([cleaned])

        # 3. Predict
        pred_id   = int(model.predict(X_vec)[0])
        scores    = get_confidence_scores(model, X_vec)
        confidence = round(scores.get(pred_id, 1.0) * 100, 1)

        # 4. Build card
        result_html = build_result_card(pred_id, scores.get(pred_id, 1.0), scores)

        # 5. Log
        logger.info(
            "Prediction → %s (%.1f%%) | input=%s",
            EMOTION_MAP[pred_id]['label'], confidence, text[:50]
        )

        # 6. Update history
        prediction_history.append({
            "id":         pred_id,
            "text":       text.strip(),
            "confidence": confidence,
        })

        # 7. Return
        return result_html, cleaned, build_history_html()

    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        return f"❌ Error during prediction: {str(exc)}", "", build_history_html()

# ── Section 11: Custom CSS ───────────────────────────────────
CUSTOM_CSS = """
/* ─── Root & body ─── */
:root {
  --color-accent:  #6366f1;
  --color-accent2: #8b5cf6;
}
body, .gradio-container {
  background: #0a0f1e !important;
  font-family: 'Segoe UI', system-ui, sans-serif !important;
}
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }

/* ─── Header banner ─── */
#header-banner {
  background: linear-gradient(135deg,#1e1b4b,#0f172a,#1a0f2e);
  border: 1px solid #312e81;
  border-radius: 18px;
  padding: 32px 40px;
  text-align: center;
  margin-bottom: 8px;
}

/* ─── Tabs ─── */
.tab-nav { background: #111827 !important; border-radius: 10px !important; }
.tab-nav button.selected {
  background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
  color: white !important; border-radius: 8px !important;
}

/* ─── Input box ─── */
textarea {
  background: #0f172a !important;
  border: 1.5px solid #1e293b !important;
  border-radius: 14px !important;
  color: #f1f5f9 !important;
  font-size: 15px !important;
  line-height: 1.6 !important;
  padding: 16px !important;
  transition: border-color .25s !important;
}
textarea:focus {
  border-color: #6366f1 !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,.15) !important;
}

/* ─── Buttons ─── */
button.primary {
  background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
  border: none !important; border-radius: 12px !important;
  font-weight: 700 !important; font-size: 15px !important;
  padding: 12px 28px !important; transition: all .25s !important;
  box-shadow: 0 4px 20px rgba(99,102,241,.35) !important;
}
button.primary:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 28px rgba(99,102,241,.5) !important;
}
button.secondary {
  background: #1e293b !important;
  border: 1px solid #334155 !important;
  border-radius: 10px !important; color: #94a3b8 !important;
  transition: all .2s !important;
}
button.secondary:hover {
  border-color: #6366f1 !important; color: #6366f1 !important;
}

/* ─── Panels / blocks ─── */
.block { background: #111827 !important; border-color: #1e293b !important; border-radius: 16px !important; }
.panel { background: #111827 !important; }
label span { color: #94a3b8 !important; font-size: 13px !important; font-weight: 600 !important; }

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
"""

# ── Section 12: Example Texts ────────────────────────────────
EXAMPLES = [
    ["I feel so heartbroken and empty inside. Nothing makes sense anymore."],
    ["This is the best day of my life! I got the job I always dreamed of!"],
    ["I love you so deeply, you mean everything to me."],
    ["How dare you do this to me! I am absolutely furious right now!"],
    ["I am terrified of what might happen next. I can't stop shaking."],
    ["Wait — what?! I cannot believe that just happened! Wow!"],
]

# ── Section 13: Build Gradio UI ─────────────────────────────
def build_ui() -> gr.Blocks:
    with gr.Blocks(
        css   = CUSTOM_CSS,
        title = "SentimentAI — Emotion Detector",
    ) as demo:

        # ── Header ──
        gr.HTML("""
        <div id="header-banner">
          <div style="font-size:48px;margin-bottom:10px;">🧠</div>
          <h1 style="font-size:32px;font-weight:800;color:#f1f5f9;
                     margin:0 0 8px;letter-spacing:-1px;">
            Sentiment<span style="background:linear-gradient(90deg,#818cf8,#c084fc);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">AI</span>
          </h1>
          <p style="color:#94a3b8;font-size:15px;margin:0;">
            Real-time emotion detection powered by Machine Learning
          </p>
          <div style="display:flex;justify-content:center;gap:10px;margin-top:16px;flex-wrap:wrap;">
            <span style="background:rgba(99,102,241,.15);border:1px solid rgba(99,102,241,.3);
                          color:#818cf8;padding:4px 14px;border-radius:20px;font-size:12px;font-weight:700;">
              ⚡ TF-IDF + Scikit-learn
            </span>
            <span style="background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.3);
                          color:#34d399;padding:4px 14px;border-radius:20px;font-size:12px;font-weight:700;">
              ✅ 6 Emotion Classes
            </span>
            <span style="background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3);
                          color:#fbbf24;padding:4px 14px;border-radius:20px;font-size:12px;font-weight:700;">
              🎯 Confidence Scores
            </span>
          </div>
        </div>""")

        # ── Main layout ──
        with gr.Row(equal_height=False):
            # Left column — input
            with gr.Column(scale=5):
                txt_input = gr.Textbox(
                    label       = "✏️  Enter your text",
                    placeholder = "Type or paste your text here… (press Ctrl+Enter to analyze)",
                    lines       = 6,
                    max_lines   = 12,
                )

                with gr.Row():
                    btn_predict = gr.Button("🔍  Analyze Emotion", variant="primary", scale=3)
                    btn_clear   = gr.Button("🗑️  Clear",           variant="secondary", scale=1)

                gr.Examples(
                    examples   = EXAMPLES,
                    inputs     = [txt_input],
                    label      = "💡 Try an example",
                    examples_per_page = 6,
                )

                # Cleaned text accordion
                with gr.Accordion("🔬 Preprocessed Text (debug)", open=False):
                    txt_cleaned = gr.Textbox(
                        label     = "Cleaned input sent to model",
                        interactive = False,
                        lines     = 2,
                    )

            # Right column — output
            with gr.Column(scale=5):
                out_card    = gr.HTML(label="Prediction Result")
                out_history = gr.HTML(label="Recent Predictions", value="<p style='color:#64748b;font-size:13px;'>No predictions yet.</p>")

        # ── Wire events ──
        btn_predict.click(
            fn      = predict,
            inputs  = [txt_input],
            outputs = [out_card, txt_cleaned, out_history],
        )
        txt_input.submit(    # Enter key
            fn      = predict,
            inputs  = [txt_input],
            outputs = [out_card, txt_cleaned, out_history],
        )
        btn_clear.click(
            fn      = lambda: ("", "", ""),
            inputs  = [],
            outputs = [txt_input, out_card, txt_cleaned],
        )

    return demo

# ── Section 14: Entry Point ──────────────────────────────────
if __name__ == "__main__":
    port   = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    host   = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    share  = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    debug  = os.environ.get("GRADIO_DEBUG", "false").lower() == "true"

    logger.info("Starting SentimentAI on %s:%s", host, port)
    demo = build_ui()
    demo.launch(
        server_name = host,
        server_port = port,
        share       = share,
        debug       = debug,
        show_error  = True,
    )
