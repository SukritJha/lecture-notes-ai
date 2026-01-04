# 🎓 AI Lecture Assistant

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![OpenAI Whisper](https://img.shields.io/badge/OpenAI_Whisper-000000?style=for-the-badge&logo=openai&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)

**Turn any lecture into a perfect study guide instantly.**

A full-stack AI application that ingests audio (YouTube/MP3), transcribes it using OpenAI's Whisper, and synthesizes it into study materials using Large Language Models (LLMs). It solves the "passive listening" problem by forcing active recall through auto-generated quizzes and flashcards.

---

## 📸 Interface Preview

*(Upload your screenshots to an 'assets' folder and link them here)*

| **Dashboard & Analysis** | **Interactive Mind Map** |
|:---:|:---:|
| <img src="https://via.placeholder.com/400x200?text=Dashboard+Screenshot" width="100%"> | <img src="https://via.placeholder.com/400x200?text=Mind+Map+Screenshot" width="100%"> |

---

## ⚡ Key Features

- **Multi-Modal Ingest:** Handles local MP3/WAV files and direct YouTube URL streaming via `yt-dlp`.
- **Hybrid AI Pipeline:**
  - **Transcription:** Runs OpenAI Whisper locally (GPU-accelerated) for privacy and zero cost.
  - **Synthesis:** Uses Google Gemini 1.5 or Llama 3 (Groq) for high-level reasoning.
- **Active Recall Tools:**
  - **Cornell Notes:** Automatically formatted summaries with key takeaways.
  - **Zoomable Mind Maps:** Hierarchical visualization of topics using `markmap`.
  - **JSON-Based Quizzes:** Structured data generation for reliable testing.
- **Multilingual Support:** Auto-detects foreign languages (e.g., Hindi, French) and translates notes to English.

## 🛠️ Installation

**Prerequisites:** Python 3.8+ and [FFmpeg](https://ffmpeg.org/download.html).

1. **Clone the Repo**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/lecture-notes-ai.git](https://github.com/YOUR_USERNAME/lecture-notes-ai.git)
   cd lecture-notes-ai
