<div align="center">

# 🧠 SentimentAI — Emotion Detection AI

### *Real-time emotion detection powered by Machine Learning*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-FF6C37?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Anmoldhiman17/emotion-detector-ai)
[![NLP](https://img.shields.io/badge/NLP-TF--IDF-4ECDC4?style=for-the-badge)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge)](https://huggingface.co/spaces/Anmoldhiman17/emotion-detector-ai)
[![Emotions](https://img.shields.io/badge/Emotions-6_Classes-BC8CFF?style=for-the-badge)](#)

<br/>

> ⚡ **Next-Gen AI** · 🧠 **NLP** · 🎭 **Real-Time Intelligence** · 🚀 **Production-Ready**

</div>

---

## 🌐 Live Demo

> 🟢 **Live and running on Hugging Face Spaces!**

🔗 **[https://huggingface.co/spaces/Anmoldhiman17/emotion-detector-ai](https://huggingface.co/spaces/Anmoldhiman17/emotion-detector-ai)**

---

## 🧠 About the Project

> *"Language is the dress of thought — SentimentAI reads between the lines so you don't have to."*

**SentimentAI** is a **next-generation emotion detection system** built with classical Machine Learning and modern UI/UX principles. It understands the emotional tone of any text input — whether it's a tweet, review, message, or essay — and classifies it into one of **6 core human emotions** in real-time.

Powered by **TF-IDF vectorization** and **Logistic Regression**, the model is trained on thousands of labeled text samples and achieves high accuracy across all emotion classes. The entire experience is wrapped in a **glassmorphism-style Gradio UI** that rivals modern SaaS AI tools.

This project demonstrates the complete ML lifecycle — from **data preprocessing** and **model training** to **serialization**, **inference**, and **production deployment** on Hugging Face Spaces.

---

## ✨ Features

| Feature | Description |
|---|---|
| ⚡ **Real-Time Prediction** | Instant emotion classification with sub-second inference |
| 📊 **Confidence Scores** | Full probability distribution across all 6 emotion classes |
| 🎭 **6 Emotion Classes** | Sadness, Joy, Love, Anger, Fear, Surprise |
| 🕒 **Prediction History** | Tracks your last 3 predictions with timestamps |
| 💡 **Example Suggestions** | One-click example sentences for each emotion |
| 🎨 **Glassmorphism UI** | Premium dark-theme with glowing cards and animations |
| 🔧 **Smart Preprocessing** | 9-step NLP pipeline fully automated |
| 📱 **Fully Responsive** | Pixel-perfect on mobile, tablet, and desktop |

---

## 🛠️ Tech Stack

| Technology | Purpose | Version |
|---|---|---|
| 🐍 Python | Core language | 3.10+ |
| 🎛️ Gradio | Web UI framework | 4.26+ |
| 🔬 Scikit-learn | ML algorithms | 1.4+ |
| 📐 TF-IDF | Text vectorization | — |
| 📈 Logistic Regression | Classification | — |
| 🔢 NumPy | Numerical ops | 1.26+ |
| 📦 Pickle | Model serialization | stdlib |
| 🌐 NLTK | NLP preprocessing | 3.8+ |
| 🤗 Hugging Face Spaces | Deployment | — |

---

## ⚙️ How It Works

```
User Input → Preprocess → TF-IDF Vectorize → LR Model → Emotion + Confidence %
```

### 🧹 Preprocessing Pipeline (9 Steps):

```
01 → Lowercase text
02 → Strip HTML tags & entities
03 → Remove URLs and hyperlinks
04 → Remove email addresses
05 → Remove punctuation & special characters
06 → Remove digits and numbers
07 → Remove English stopwords (NLTK)
08 → Collapse extra whitespace
09 → TF-IDF vectorize → LR predict → softmax probabilities
```

---

## 🎭 Emotion Labels

| ID | Emotion | Emoji | Color |
|---|---|---|---|
| 0 | Sadness | 😢 | `#6495ED` |
| 1 | Joy | 😊 | `#FFD700` |
| 2 | Love | ❤️ | `#FF69B4` |
| 3 | Anger | 😡 | `#FF4500` |
| 4 | Fear | 😨 | `#9370DB` |
| 5 | Surprise | 😲 | `#FFA500` |

---

## 📊 Model Performance

| Emotion | Precision | Recall | F1-Score |
|---|---|---|---|
| 😢 Sadness | 0.89 | 0.91 | 0.90 |
| 😊 Joy | 0.93 | 0.95 | 0.94 |
| ❤️ Love | 0.88 | 0.87 | 0.87 |
| 😡 Anger | 0.85 | 0.84 | 0.85 |
| 😨 Fear | 0.91 | 0.89 | 0.90 |
| 😲 Surprise | 0.83 | 0.82 | 0.82 |
| **Overall** | — | — | **~89%** |

---

## 🖼️ Screenshots

> 📸 *Add your screenshots here*

![App Screenshot](screenshots/main.png)
![Result Card](screenshots/result.png)
![Mobile View](screenshots/mobile.png)

---

## 🎯 Use Cases

- 💬 **Social Media Monitoring** — Analyze tweets, posts, and comments at scale
- ⭐ **Customer Review Analysis** — Classify feedback and identify sentiment patterns
- 🧘 **Mental Health Monitoring** — Detect sadness or fear in journal entries
- 🤖 **Emotion-Aware Chatbots** — Help AI respond with empathy
- 📚 **Education & Research** — Complete NLP portfolio project
- 📰 **News & Media Analysis** — Identify emotional tone in articles

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/Anmoldhiman17/emotion-detector-ai.git
cd emotion-detector-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place model files & run
# → Add model.pkl and vectorizer.pkl to root
python app.py
# → Open http://localhost:7860
```

**`requirements.txt`:**
```
gradio>=4.26.0
scikit-learn>=1.4.0
numpy>=1.26.0
pandas>=2.2.0
nltk>=3.8.1
scipy>=1.12.0
```

---

## ☁️ Deployment

### 🤗 Hugging Face Spaces (Recommended — Free)

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → Create new Space
2. Select **Gradio** as the SDK
3. Upload `app.py`, `model.pkl`, `vectorizer.pkl`, `requirements.txt`
4. HF auto-builds → Live in ~2 minutes 🎉

### 🚂 Railway / Render Alternative

Connect GitHub repo → Set Start Command: `python app.py` → Set `PORT=7860` → Deploy!

---

## 🔮 Future Improvements

- [ ] 🤗 Upgrade to BERT / RoBERTa transformer model `[Soon]`
- [ ] 🌍 Multilingual support — 10+ languages `[Planned]`
- [ ] 📊 Analytics dashboard with emotion trend charts `[Planned]`
- [ ] 🔌 REST API endpoint for third-party integration `[Soon]`
- [ ] 🎤 Voice input — analyze emotion from speech `[Idea]`
- [ ] 📱 Native mobile app (React Native / Flutter) `[Idea]`
- [ ] 📦 Batch CSV processing `[Planned]`
- [ ] 🔐 User accounts + saved prediction history `[Idea]`

---

## 🤝 Contribution

Contributions, issues, and feature requests are **welcome!**

```bash
git checkout -b feature/AmazingFeature
git commit -m "Add: AmazingFeature"
git push origin feature/AmazingFeature
# → Open a Pull Request 🎉
```

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.
Free to use, modify, and distribute — attribution appreciated 🙏

---

<div align="center">

## 🧠 Developed with passion by Anmol 🚀

*"Building intelligent systems, one model at a time."*

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Deployed on HuggingFace](https://img.shields.io/badge/Deployed%20on-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Anmoldhiman17/emotion-detector-ai)
[![by Anmol](https://img.shields.io/badge/by-Anmol-BC8CFF?style=for-the-badge)](https://github.com/Anmoldhiman17)

*If this project helped you, please consider giving it a ⭐ on GitHub!*

© 2024 Anmol · SentimentAI · MIT License

</div>
