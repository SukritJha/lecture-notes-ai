import streamlit as st
import os
import yt_dlp
import whisper
import google.generativeai as genai
import json
import re
import html
import time
import torch
from fpdf import FPDF
from google.api_core.exceptions import ResourceExhausted
from streamlit_markmap import markmap 

# Try to import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Lecture Assistant", layout="wide", page_icon="🎓")

# --- CSS GENERATOR ---
def get_css():
    main_bg = "#0f172a"
    surface = "#1e293b" 
    text_primary = "#e5e7eb"
    text_secondary = "#9ca3af"
    accent = "#3b82f6"
    card_front = surface
    card_back = accent
    card_border = "#334155"
    btn_text = "#ffffff"

    return f"""
    <style>
        .stApp {{ background-color: {main_bg}; }}
        h1, h2, h3, h4, h5, h6, p, li, span, div.stMarkdown, .stText, label {{ color: {text_primary} !important; }}
        .stCaption, .small-text {{ color: {text_secondary} !important; }}
        
        div.stButton > button, div.stDownloadButton > button, div.stFormSubmitButton > button {{ 
            background-color: {accent} !important; color: {btn_text} !important; border: none; padding: 10px 24px; border-radius: 8px; font-weight: 600; transition: 0.3s; box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        }}
        div.stButton > button:hover, div.stDownloadButton > button:hover, div.stFormSubmitButton > button:hover {{ 
            opacity: 0.9; transform: translateY(-1px); box-shadow: 0 4px 6px rgba(0,0,0,0.15); color: {btn_text} !important; 
        }}
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {{ background-color: {surface} !important; color: {text_primary} !important; border: 1px solid {card_border}; }}

        /* HEADER */
        .main-header {{ font-family: 'Helvetica Neue', sans-serif; text-align: center; padding: 2rem 0; background: linear-gradient(180deg, {surface} 0%, {main_bg} 100%); border-bottom: 1px solid {card_border}; margin-bottom: 2rem; border-radius: 0 0 16px 16px; }}
        .main-header h1 {{ font-size: 2.5rem; font-weight: 700; color: {accent} !important; margin: 0; }}
        .main-header p {{ font-size: 1.1rem; color: {text_secondary} !important; margin-top: 0.5rem; }}
        
        /* CARDS & LINKS */
        .flashcard-container {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; padding: 20px; }}
        .flip-card {{ background-color: transparent; width: 300px; height: 200px; perspective: 1000px; }}
        .flip-card-inner {{ position: relative; width: 100%; height: 100%; text-align: center; transition: transform 0.6s; transform-style: preserve-3d; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); border-radius: 12px; }}
        .flip-card:hover .flip-card-inner {{ transform: rotateY(180deg); }}
        .flip-card-front, .flip-card-back {{ position: absolute; width: 100%; height: 100%; -webkit-backface-visibility: hidden; backface-visibility: hidden; border-radius: 12px; display: flex; align-items: center; justify-content: center; padding: 24px; flex-direction: column; }}
        .flip-card-front {{ background-color: {card_front}; color: {text_primary}; border: 1px solid {card_border}; }}
        .flip-card-back {{ background-color: {card_back}; color: #ffffff; transform: rotateY(180deg); font-size: 15px; line-height: 1.6; overflow-y: auto; }}
        .rec-link {{ display: block; padding: 10px; margin: 5px 0; background-color: {surface}; border: 1px solid {card_border}; border-radius: 8px; color: {text_primary} !important; text-decoration: none; transition: 0.2s; }}
        .rec-link:hover {{ border-color: {accent}; background-color: {main_bg}; }}
        
        /* FOOTER */
        .footer {{ position: fixed; left: 0; bottom: 0; width: 100%; background-color: {surface}; color: {text_secondary}; text-align: center; padding: 10px; font-size: 13px; border-top: 1px solid {card_border}; z-index: 100; }}
        .block-container {{ padding-bottom: 80px; }}
    </style>
    """
st.markdown(get_css(), unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
    <div class="main-header">
        <h1>🎓 AI Lecture Assistant</h1>
        <p>Turn any video into detailed notes, quizzes, mind maps & flashcards instantly.</p>
    </div>
""", unsafe_allow_html=True)

# --- GLOBAL SESSION STATE ---
if "gemini_model" not in st.session_state: st.session_state.gemini_model = None
if "groq_client" not in st.session_state: st.session_state.groq_client = None

# --- UNIFIED AI CALLER ---
def generate_ai_response(prompt, provider, mock_mode=False):
    if mock_mode:
        time.sleep(1.0)
        return "MOCK DATA: The AI is simulating a response here because Mock Mode is enabled in settings."
    
    try:
        if provider == "Gemini (Google)":
            if not st.session_state.gemini_model: raise Exception("Gemini API Key not set.")
            response = st.session_state.gemini_model.generate_content(prompt)
            return response.text
        elif provider == "Groq (Llama 3)":
            if not GROQ_AVAILABLE: return "Error: Groq library not installed."
            if not st.session_state.groq_client: raise Exception("Groq API Key not set.")
            chat_completion = st.session_state.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile",
            )
            return chat_completion.choices[0].message.content
    except ResourceExhausted: return "Error: Quota Exceeded. Switch provider!"
    except Exception as e: return f"Error: {str(e)}"
    return "Error: No provider selected."

# --- CORE LOGIC ---

@st.cache_resource
def load_whisper_model():
    # Only loads the model. No printing to UI.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return whisper.load_model("base", device=device)

def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best', 
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}], 
        'outtmpl': 'temp_audio.%(ext)s', 
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([url])
    return "temp_audio.mp3"

@st.cache_data
def transcribe_audio(_model, audio_path, unique_id, translate=False, mock_mode=False):
    if mock_mode: return "This is a mock transcript of a lecture about Artificial Intelligence. The professor discusses neural networks, backpropagation, and the importance of data quality."
    
    task = "translate" if translate else "transcribe"
    result = _model.transcribe(audio_path, task=task)
    return result["text"]

def parse_json(text):
    match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if match: text_content = match.group(0)
    else:
        text_content = re.sub(r"```json", "", text)
        text_content = re.sub(r"```", "", text_content)
    try: return json.loads(text_content)
    except: return None

class PDF(FPDF):
    def header(self): pass
    def chapter_title(self, label):
        self.set_font('Arial', 'B', 16); self.set_text_color(0, 128, 128) 
        self.cell(0, 10, label, ln=True, align='L'); self.set_draw_color(0, 128, 128); self.line(self.get_x(), self.get_y(), 190, self.get_y()); self.ln(4)
    def chapter_body(self, body):
        self.set_font('Arial', '', 12); self.set_text_color(0, 0, 0); self.multi_cell(0, 8, body); self.ln()
    def bullet_point(self, text):
        self.set_font('Arial', '', 12); self.set_text_color(0, 0, 0); self.set_x(15); self.multi_cell(0, 8, f"{chr(149)} {text}")

def create_styled_pdf(markdown_text):
    if not markdown_text: return None
    pdf = PDF(); pdf.add_page(); pdf.set_auto_page_break(auto=True, margin=15)
    def clean(t): return t.encode('latin-1', 'replace').decode('latin-1')
    try:
        lines = markdown_text.split('\n')
        for line in lines:
            line = clean(line.strip())
            if not line: continue 
            if line.startswith('#'): pdf.chapter_title(line.replace('#', '').strip())
            elif line.startswith('* ') or line.startswith('- '): pdf.bullet_point(line[2:].replace('**', ''))
            else: pdf.chapter_body(line.replace('**', ''))
        return pdf.output(dest='S').encode('latin-1')
    except: return None

# --- HTML GENERATOR FOR FULLSCREEN MINDMAP (WITH WHITE TEXT FIX) ---
def create_fullscreen_html(markdown_content):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Mind Map Fullscreen</title>
        <style>
            body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; background-color: #0f172a; overflow: hidden; }}
            svg {{ width: 100%; height: 100%; background-color: #0f172a; }}
            
            /* --- NUCLEAR CSS TO FORCE WHITE TEXT --- */
            .markmap-node text {{ fill: #ffffff !important; color: #ffffff !important; }}
            .markmap-node tspan {{ fill: #ffffff !important; color: #ffffff !important; }}
            .markmap-foreign {{ color: #ffffff !important; }}
            div {{ color: #ffffff !important; }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
        <script src="https://cdn.jsdelivr.net/npm/markmap-view@0.15.4"></script>
        <script src="https://cdn.jsdelivr.net/npm/markmap-lib@0.15.4/dist/browser/index.min.js"></script>
    </head>
    <body>
        <svg id="mindmap"></svg>
        <script>
            const markdown = `{markdown_content.replace('`', '\`')}`;
            const {{ Transformer }} = window.markmap;
            const transformer = new Transformer();
            const {{ root, features }} = transformer.transform(markdown);
            const {{ Markmap }} = window.markmap;
            Markmap.create("#mindmap", {{
                colorFrozen: true,
                initialExpandLevel: 2,
            }}, root);
        </script>
    </body>
    </html>
    """
    return html_content

# --- CALLBACKS ---
def generate_quiz_callback():
    if st.session_state.transcript:
        with st.spinner("🤖 Crafting questions..."):
            num_q = st.session_state.num_questions_slider
            provider = st.session_state.active_provider
            mock = st.session_state.mock_mode
            
            prompt = f"Create {num_q} quiz questions (JSON format only, no extra text): [{{'question':'...','options':['...'],'answer_index':0}}]\nTRANSCRIPT: {st.session_state.transcript[:30000]}"
            res_text = generate_ai_response(prompt, provider, mock)
            
            # --- DEBUG INFO IN SIDEBAR ---
            with st.sidebar:
                with st.expander("🛠️ Debug: Last AI Response"):
                    st.text(res_text[:500]) 

            if mock:
                data = [{"question": "What is AI?", "options": ["Magic", "Computer Systems", "Food"], "answer_index": 1}]
            else:
                data = parse_json(res_text)
            
            if data:
                st.session_state.quiz_data = data
                st.session_state.quiz_error = None 
            else:
                st.session_state.quiz_error = "Failed to parse Quiz JSON. Check Sidebar Debug."

def generate_flashcards_callback():
    if st.session_state.transcript:
        with st.spinner("⚡ Extracting concepts..."):
            provider = st.session_state.active_provider
            mock = st.session_state.mock_mode
            
            prompt = f"Identify 8 key terms. Return ONLY JSON list (no extra text): [{{'term': 'Concept', 'definition': 'Short definition'}}]\nTRANSCRIPT: {st.session_state.transcript[:30000]}"
            res_text = generate_ai_response(prompt, provider, mock)
            
            if mock:
                data = [{"term": "Neural Net", "definition": "A computer system modeled on the human brain."}]
            else:
                data = parse_json(res_text)
            
            if data:
                st.session_state.flashcards = data
                st.session_state.card_error = None
            else:
                st.session_state.card_error = "Failed to generate cards. Try again."

def generate_mindmap_callback():
    if st.session_state.transcript:
        with st.spinner("🧠 Brainstorming connections..."):
            provider = st.session_state.active_provider
            mock = st.session_state.mock_mode
            
            # IMPROVED PROMPT: Forces brevity for cleaner maps
            prompt = f"""
            Create a hierarchical mind map. 
            RULES:
            1. Use Markdown list format (- Item).
            2. Keep nodes extremely short (max 4 words).
            3. Max depth of 3 levels.
            4. Focus on keywords, not sentences.
            
            Example:
            # AI
            ## Machine Learning
            - Supervised
            - Unsupervised
            ## Neural Networks
            - Layers
            - Weights
            
            TRANSCRIPT: {st.session_state.transcript[:30000]}
            """
            
            res_text = generate_ai_response(prompt, provider, mock)
            
            if mock:
                res_text = "# Artificial Intelligence\n## Machine Learning\n- Supervised\n- Unsupervised\n## Neural Networks\n- Layers\n- Activation Functions"
            
            if res_text:
                st.session_state.mindmap_data = res_text
                st.session_state.mindmap_error = None
            else:
                st.session_state.mindmap_error = "Failed to generate mind map."

# --- UI SETUP ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.session_state.mock_mode = st.toggle("🛠️ Mock Mode (Free/Test)", value=True, help="Use dummy data to test UI without using API credits.")
    translate_mode = st.toggle("🌍 Force English Output", value=False, help="Translates non-English audio to English text.")
    
    st.divider()
    
    st.subheader("🤖 AI Model Provider")
    provider = st.selectbox("Select Active Model:", ["Gemini (Google)", "Groq (Llama 3)"], key="active_provider")
    
    if not st.session_state.mock_mode:
        if provider == "Gemini (Google)":
            gemini_key = st.text_input("Gemini API Key", type="password")
            if gemini_key:
                genai.configure(api_key=gemini_key)
                st.session_state.gemini_model = genai.GenerativeModel('models/gemini-flash-latest')
                st.success("Gemini Ready! ✅")
        elif provider == "Groq (Llama 3)":
            if not GROQ_AVAILABLE: st.error("⚠️ `groq` library missing. Run: pip install groq")
            else:
                groq_key = st.text_input("Groq API Key", type="password")
                if groq_key:
                    st.session_state.groq_client = Groq(api_key=groq_key)
                    st.success("Groq Ready! ✅")
    
    st.divider()
    
    if torch.cuda.is_available():
        st.success("🚀 GPU Detected! Transcription will be fast.")
    else:
        st.warning("⚠️ No GPU detected. Using CPU (slower).")

    if st.button("🔄 Reset App", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- MAIN SOURCE SELECTION ---
st.write("### 📂 Source Material")
col1, col2 = st.columns([1, 2])
with col1: source_mode = st.radio("Select Input:", ["YouTube Link", "Remote File URL", "Local File Upload"], label_visibility="collapsed")
with col2:
    url = ""; uploaded_file = None
    if source_mode == "YouTube Link": url = st.text_input("YouTube URL:", placeholder="https://youtube.com/...", label_visibility="collapsed")
    elif source_mode == "Remote File URL": url = st.text_input("File URL:", placeholder="https://example.com/lecture.mp3", label_visibility="collapsed")
    elif source_mode == "Local File Upload": uploaded_file = st.file_uploader("Upload File:", type=['mp3', 'wav', 'm4a', 'mp4'], label_visibility="collapsed")

# Initialize Session State
if "transcript" not in st.session_state: st.session_state.transcript = None
if "notes" not in st.session_state: st.session_state.notes = None
if "flashcards" not in st.session_state: st.session_state.flashcards = None
if "quiz_data" not in st.session_state: st.session_state.quiz_data = None
if "mindmap_data" not in st.session_state: st.session_state.mindmap_data = None
if "recommendations" not in st.session_state: st.session_state.recommendations = None
if "quiz_error" not in st.session_state: st.session_state.quiz_error = None
if "card_error" not in st.session_state: st.session_state.card_error = None
if "mindmap_error" not in st.session_state: st.session_state.mindmap_error = None
if "nav_selection" not in st.session_state: st.session_state.nav_selection = "📚 Notes"

if st.button("🚀 Process Lecture", use_container_width=True):
    ready = False
    if st.session_state.mock_mode: ready = True
    elif provider == "Gemini (Google)" and st.session_state.gemini_model: ready = True
    elif provider == "Groq (Llama 3)" and st.session_state.groq_client: ready = True
    
    input_ready = False
    unique_id = None
    if source_mode == "Local File Upload" and uploaded_file:
        input_ready = True; unique_id = uploaded_file.name + str(uploaded_file.size)
    elif url: input_ready = True; unique_id = url

    if not ready: st.error(f"⛔ STOP: Please enter the API Key for {provider} or enable Mock Mode.")
    elif not input_ready: st.error("⛔ STOP: Please provide a URL or upload a file.")
    else:
        with st.status("Processing...", expanded=True) as status:
            target_audio_path = "temp_audio.mp3"
            
            # 1. AUDIO AQUISITION
            if not st.session_state.mock_mode:
                try:
                    if source_mode == "Local File Upload":
                        st.write("📥 Processing Uploaded File...")
                        with open(target_audio_path, "wb") as f: f.write(uploaded_file.getbuffer())
                    else:
                        st.write(f"📥 Downloading from {source_mode}...")
                        target_audio_path = download_audio(url)
                    
                    st.write("🎧 Transcribing & Translating...")
                    whisper_model = load_whisper_model()
                    
                    transcript_text = transcribe_audio(whisper_model, target_audio_path, unique_id, translate=translate_mode, mock_mode=False)
                    
                except Exception as e:
                    st.error(f"Error getting audio: {e}"); status.update(label="Failed", state="error"); st.stop()
            else:
                time.sleep(1); transcript_text = transcribe_audio(None, None, None, mock_mode=True)
            
            st.session_state.transcript = transcript_text
            
            # 2. GENERATE NOTES (Cornell Style)
            st.write(f"🤖 Generating Notes using {provider}...")
            prompt = f"Create detailed study notes using Cornell Method structure. Use '# Heading' and '- Bullet' format.\nTRANSCRIPT: {transcript_text[:30000]}"
            response_text = generate_ai_response(prompt, provider, st.session_state.mock_mode)
            st.session_state.notes = response_text
            
            # 3. GENERATE RECOMMENDATIONS
            st.write("🔗 Curating recommendations...")
            rec_prompt = f"""Based on the transcript, recommend 3 distinct topics for further reading and 3 specific search queries for related videos. Return ONLY a JSON object in this format (no extra text): {{"blogs": ["Topic 1", "Topic 2", "Topic 3"], "videos": ["Search Query 1", "Search Query 2", "Search Query 3"]}}\nTRANSCRIPT: {transcript_text[:30000]}"""
            rec_response = generate_ai_response(rec_prompt, provider, st.session_state.mock_mode)
            if st.session_state.mock_mode:
                st.session_state.recommendations = {"blogs": ["Deep Learning", "Neural Networks"], "videos": ["3Blue1Brown Neural Networks"]}
            else:
                st.session_state.recommendations = parse_json(rec_response)

            st.session_state.flashcards = None
            st.session_state.quiz_data = None
            st.session_state.mindmap_data = None
            status.update(label="Done!", state="complete", expanded=False)

# --- RESULTS DISPLAY ---
if st.session_state.transcript:
    st.divider()
    video_col, content_col = st.columns([2, 3]) 
    
    with video_col:
        st.subheader("📺 Original Source")
        if source_mode == "YouTube Link" and url: st.video(url)
        elif source_mode == "Local File Upload" and uploaded_file: st.audio(uploaded_file)
        elif source_mode == "Remote File URL" and url: st.audio(url)
        
        if st.session_state.recommendations:
            st.divider(); st.subheader("🔍 Recommended Resources")
            st.markdown("**📚 Further Reading**")
            for topic in st.session_state.recommendations.get('blogs', []):
                search_url = f"https://www.google.com/search?q={topic.replace(' ', '+')}"
                st.markdown(f"""<a href="{search_url}" target="_blank" class="rec-link">📖 {topic}</a>""", unsafe_allow_html=True)
            
            st.markdown("**🎥 Related Videos**")
            for query in st.session_state.recommendations.get('videos', []):
                search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
                st.markdown(f"""<a href="{search_url}" target="_blank" class="rec-link">▶️ {query}</a>""", unsafe_allow_html=True)

    with content_col:
        selection = st.radio("Navigation", ["📚 Notes", "💬 Chat", "📝 Quiz", "⚡ Flashcards", "🧠 Mind Map"], horizontal=True, label_visibility="collapsed", key="nav_selection")
        st.divider()

        if selection == "📚 Notes":
            if st.session_state.notes:
                st.markdown(st.session_state.notes)
                pdf_bytes = create_styled_pdf(st.session_state.notes)
                if pdf_bytes: st.download_button("Download PDF", pdf_bytes, "notes.pdf", "application/pdf")
        
        elif selection == "💬 Chat":
            if "messages" not in st.session_state: st.session_state.messages = []
            for m in st.session_state.messages:
                with st.chat_message(m["role"]): st.markdown(m["content"])
            if q := st.chat_input("Ask a question..."):
                st.session_state.messages.append({"role":"user", "content":q})
                with st.chat_message("user"): st.markdown(q)
                ans = generate_ai_response(f"Answer based on transcript: {st.session_state.transcript[:30000]}\nQ: {q}", provider, st.session_state.mock_mode)
                st.session_state.messages.append({"role":"assistant", "content":ans})
                with st.chat_message("assistant"): st.markdown(ans)
        
        elif selection == "📝 Quiz":
            st.slider("Questions:", 1, 10, 5, key="num_questions_slider")
            st.button("Generate Quiz", on_click=generate_quiz_callback)
            if st.session_state.quiz_error: st.error(st.session_state.quiz_error)
            
            if st.session_state.quiz_data:
                with st.form("quiz"):
                    score = 0; user_answers = {}
                    for i, q in enumerate(st.session_state.quiz_data):
                        st.write(f"**Q{i+1}: {q['question']}**")
                        user_answers[i] = st.radio("Select:", q['options'], key=f"q{i}", index=None)
                        st.divider()
                    if st.form_submit_button("Submit"):
                        for i, q in enumerate(st.session_state.quiz_data):
                            if user_answers[i] == q['options'][q['answer_index']]: score += 1
                        st.success(f"🏆 Final Score: {score} / {len(st.session_state.quiz_data)}")
                        
                        st.markdown("### 📝 Detailed Review")
                        for i, q in enumerate(st.session_state.quiz_data):
                            correct = q['options'][q['answer_index']]
                            if user_answers[i] == correct: st.success(f"Q{i+1}: Correct!")
                            else: st.error(f"Q{i+1}: You chose {user_answers[i]}. Correct: {correct}")
        
        elif selection == "⚡ Flashcards":
            st.button("Generate Cards", on_click=generate_flashcards_callback)
            if st.session_state.card_error: st.error(st.session_state.card_error)
            if st.session_state.flashcards:
                cards_html = '<div class="flashcard-container">'
                for card in st.session_state.flashcards:
                    term = html.escape(card['term']); defn = html.escape(card['definition'])
                    cards_html += f'<div class="flip-card"><div class="flip-card-inner"><div class="flip-card-front"><h3>{term}</h3></div><div class="flip-card-back"><p>{defn}</p></div></div></div>'
                cards_html += '</div>'
                st.markdown(cards_html, unsafe_allow_html=True)
                
        elif selection == "🧠 Mind Map":
            st.button("Generate Mind Map", on_click=generate_mindmap_callback)
            if st.session_state.mindmap_error: st.error(st.session_state.mindmap_error)
            
            if st.session_state.mindmap_data:
                # 1. Show Embedded Map (Taller height)
                markmap(st.session_state.mindmap_data, height=600)
                
                # 2. Show Fullscreen Button (Downloads HTML with WHITE TEXT fix)
                html_map = create_fullscreen_html(st.session_state.mindmap_data)
                st.download_button(
                    label="🖥️ Open Fullscreen (Interactive)",
                    data=html_map,
                    file_name="mindmap_fullscreen.html",
                    mime="text/html",
                    help="Download this file and open it in your browser for a true fullscreen experience with white text."
                )

# --- FOOTER ---
st.markdown("""<div class="footer"><p>Developed with ❤️ by Sukrit Jha</p></div>""", unsafe_allow_html=True)