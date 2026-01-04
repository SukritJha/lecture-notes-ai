import os
import yt_dlp
import whisper
import google.generativeai as genai
import textwrap

# --- CONFIGURATION (ENTER YOUR KEYS HERE) ---
# Get Key: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = "AIzaSyAuW9DUfPGrZhjIAzlBuVtbTz5AdROIJ9g" 

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('models/gemini-flash-latest')
def step1_download_video(youtube_url):
    print(f"\n[1/4] Downloading audio from YouTube: {youtube_url}...")
    
    # Options to download ONLY audio (faster)
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'downloaded_audio.%(ext)s',
        'quiet': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        print("Success: Audio downloaded as 'downloaded_audio.mp3'")
        return "downloaded_audio.mp3"
    except Exception as e:
        print(f"Error downloading: {e}")
        return None

def step2_transcribe(audio_path):
    print(f"\n[2/4] Transcribing audio on RTX 3050 (This may take a moment)...")
    
    # Load Whisper on GPU
    try:
        whisper_model = whisper.load_model("small", device="cuda")
    except RuntimeError:
        print("GPU busy or not found, falling back to CPU (slower)...")
        whisper_model = whisper.load_model("small", device="cpu")

    result = whisper_model.transcribe(audio_path)
    text = result["text"]
    
    # Save raw transcript just in case
    with open("transcript_raw.txt", "w", encoding="utf-8") as f:
        f.write(text)
        
    print(f"Success: Transcript length is {len(text)} characters.")
    return text

def step3_generate_content(transcript_text):
    print(f"\n[3/4] Sending to Gemini for Notes & Quiz generation...")
    
    # 3A: Generate Notes
    notes_prompt = f"""
    You are an expert university professor. Create detailed STUDY NOTES from this transcript.
    - Use HTML formatting (<h1>, <h2>, <ul>, <li>, <strong>).
    - Structure it with: "Introduction", "Key Concepts", "Detailed Analysis", "Summary".
    - Ignore filler words.
    
    TRANSCRIPT:
    {transcript_text[:30000]}  # Limit text to avoid errors on huge videos
    """
    
    response_notes = model.generate_content(notes_prompt)
    notes_html = response_notes.text
    
    # 3B: Generate Quiz
    quiz_prompt = f"""
    Create a Short Quiz (5 Questions) based on this transcript.
    - Format as HTML.
    - Provide the Question, Options, and the Correct Answer hidden in a <details> tag.
    
    TRANSCRIPT:
    {transcript_text[:30000]}
    """
    
    response_quiz = model.generate_content(quiz_prompt)
    quiz_html = response_quiz.text
    
    return notes_html, quiz_html

def step4_create_report(notes_html, quiz_html):
    print(f"\n[4/4] Compiling Final Report...")
    
    final_html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: auto; padding: 20px; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
            h2 {{ color: #16a085; }}
            .quiz-section {{ background-color: #f9f9f9; padding: 20px; border-left: 5px solid #f39c12; margin-top: 30px; }}
            details {{ margin-bottom: 10px; cursor: pointer; color: #555; }}
            summary {{ font-weight: bold; }}
        </style>
    </head>
    <body>
        {notes_html}
        
        <div class="quiz-section">
            <h1>Self-Assessment Quiz</h1>
            {quiz_html}
        </div>
    </body>
    </html>
    """
    
    with open("Study_Guide.html", "w", encoding="utf-8") as f:
        f.write(final_html)
    
    print("DONE! Open 'Study_Guide.html' in your browser.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Ask User for Input
    print("--- AI LECTURE NOTE GENERATOR ---")
    choice = input("Enter '1' for YouTube Link or '2' for Local File: ")
    
    audio_file = None
    
    if choice == '1':
        url = input("Paste YouTube URL: ")
        audio_file = step1_download_video(url)
    elif choice == '2':
        audio_file = input("Enter file name (e.g., lecture.mp3): ")
    
    if audio_file and os.path.exists(audio_file):
        # 2. Transcribe
        transcript = step2_transcribe(audio_file)
        
        # 3. Intelligence
        notes, quiz = step3_generate_content(transcript)
        
        # 4. Save
        step4_create_report(notes, quiz)
    else:
        print("Error: Audio file processing failed.")