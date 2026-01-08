# 🎓 AI Lecture Assistant

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![OpenAI Whisper](https://img.shields.io/badge/OpenAI_Whisper-000000?style=for-the-badge&logo=openai&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)

> **Problem:** Students consume hours of video lectures but struggle to revise, recall, and test their understanding efficiently.


**Turn any lecture into a perfect study guide instantly.**

AI Lecture Assistant is an intelligent study companion that converts long video or audio lectures into **structured notes, quizzes, flashcards, and visual mind maps**.

It eliminates passive learning by transforming lectures into **active recall materials**, helping students revise faster and retain concepts better.


---

## 📸 Interface Preview

*(Upload your screenshots to an 'assets' folder and link them here)*

| **Dashboard & Analysis** | **Interactive Mind Map** |
|:---:|:---:|
| <img src="https://via.placeholder.com/400x200?text=Dashboard+Screenshot" width="100%"> | <img src="https://via.placeholder.com/400x200?text=Mind+Map+Screenshot" width="100%"> |

---

## ⚡ Key Features

- **Multi-Source Input**
  - YouTube links, MP3/WAV, and uploaded videos

- **Offline Transcription**
  - Whisper runs locally (GPU-accelerated if available)
  - No audio sent to external servers

- **Hybrid AI Reasoning**
  - Gemini (Google) or LLaMA 3 (Groq) for summarization and Q&A

- **Active Recall Tools**
  - Cornell-style notes
  - Auto-generated quizzes
  - Flip-style flashcards
  - Interactive mind maps

- **Export Options**
  - Downloadable PDFs
  - Fullscreen interactive mind maps

- **Multilingual Support**
  - Auto-detects language and translates to English

## 🧠 How It Works (System Flow)

1. **Input Source**
   - YouTube lecture link or local audio/video file

2. **Speech Processing**
   - Audio is extracted and transcribed using OpenAI Whisper  
   - Automatically detects GPU and falls back to CPU if unavailable

3. **AI Understanding**
   - Transcript is processed by Gemini or LLaMA (Groq) for reasoning

4. **Study Material Generation**
   - Cornell-style notes
   - Auto-generated quizzes
   - Concept flashcards
   - Hierarchical mind maps

5. **Interactive Output**
   - Web-based Streamlit interface
   - PDF export, quizzes, and fullscreen mind maps


## 🛠 Installation (Local)

**Requirements**
- Python 3.8+
- FFmpeg

```bash
git clone https://github.com/YOUR_USERNAME/AI-Lecture-Assistant.git
cd AI-Lecture-Assistant
pip install -r requirements.txt
streamlit run app.py
```
## ⚠️ Limitations

- Free deployments may block YouTube downloads due to network restrictions
- Large videos increase processing time on CPU
- Real-time transcription depends on hardware availability

## 🚀 Future Enhancements

- Cloud-based transcription for faster processing
- Timestamped notes linked to video playback
- User accounts and saved lecture history
- Mobile-friendly interface
