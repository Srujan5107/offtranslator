import streamlit as st
import sqlite3
import pyttsx3
import os
import torch
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from streamlit_mic_recorder import mic_recorder
from datetime import datetime
from gtts import gTTS
import speech_recognition as sr
from deep_translator import GoogleTranslator

# ----------------- 1. DATABASE & ENGINES -----------------
def init_local_db():
    conn = sqlite3.connect('global_translator.db')
    conn.execute('CREATE TABLE IF NOT EXISTS logs (user TEXT, src_txt TEXT, res_txt TEXT, pair TEXT, mode TEXT, time TEXT)')
    conn.commit()
    conn.close()

@st.cache_resource
def load_offline_engines():
    with st.status("Initializing Global AI Engines (Offline)...", expanded=False):
        stt = WhisperModel("base", device="cpu", compute_type="int8") 
        model_id = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return stt, tokenizer, model

# ----------------- 2. UNIVERSAL LOGIC -----------------
def perform_translation(text, src_code, tgt_code, mode, nllb_tok=None, nllb_model=None):
    if not text.strip(): return ""
    if mode == "Online":
        # Google uses 2-letter codes; we slice 'eng_Latn' to 'en'
        online_code = tgt_code.split('_')[0][:2]
        return GoogleTranslator(source='auto', target=online_code).translate(text)
    else:
        nllb_tok.src_lang = src_code
        inputs = nllb_tok(text, return_tensors="pt")
        tgt_id = nllb_tok.convert_tokens_to_ids(tgt_code)
        tokens = nllb_model.generate(**inputs, forced_bos_token_id=tgt_id, max_length=150)
        return nllb_tok.batch_decode(tokens, skip_special_tokens=True)[0]

def perform_tts(text, lang_code, mode):
    if not text.strip(): return
    if mode == "Online":
        online_code = lang_code.split('_')[0][:2]
        tts = gTTS(text=text, lang=online_code)
        tts.save("temp_speech.mp3")
        st.audio("temp_speech.mp3", format="audio/mp3", autoplay=True)
    else:
        engine = pyttsx3.init()
        engine.save_to_file(text, "offline_speech.wav")
        engine.runAndWait()
        st.audio("offline_speech.wav", format="audio/wav", autoplay=True)

# ----------------- 3. UI & LANGUAGE MAP -----------------
st.set_page_config(page_title="Global Offline Translator", layout="wide")
init_local_db()

# Expanded Language Dictionary (200+ are possible, here are the most common)
ALL_LANGS = {
    # Indian
    "Hindi": "hin_Deva", "Bengali": "ben_Beng", "Tamil": "tam_Taml", "Telugu": "tel_Telu", 
    "Marathi": "mar_Deva", "Gujarati": "guj_Gujr", "Kannada": "kan_Knda", "Malayalam": "mal_Mlym",
    "Punjabi": "pan_Guru", "Urdu": "urd_Arab", "Assamese": "asm_Beng", "Odia": "ory_Orya",
    # Global
    "English": "eng_Latn", "Spanish": "spa_Latn", "French": "fra_Latn", "German": "deu_Latn",
    "Chinese (Simplified)": "zho_Hans", "Japanese": "jpn_Jpan", "Korean": "kor_Hang",
    "Russian": "rus_Cyrl", "Arabic": "arb_Arab", "Portuguese": "por_Latn", "Italian": "ita_Latn",
    "Turkish": "tur_Latn", "Vietnamese": "vie_Latn", "Thai": "tha_Thai"
}

if "manual_out" not in st.session_state: st.session_state.manual_out = ""

# Sidebar
with st.sidebar:
    st.title("Settings")
    current_mode = "Online" if st.toggle("🌐 Online Mode", value=False) else "Offline"
    st.info(f"Mode: **{current_mode}**")
    st.divider()
    st.write("### Translation History")
    # Quick history view
    conn = sqlite3.connect('global_translator.db')
    history = conn.execute("SELECT res_txt FROM logs ORDER BY time DESC LIMIT 3").fetchall()
    for h in history: st.caption(f"Last: {h[0][:40]}...")

# Engine Load
n_tok, n_mod, s_eng = None, None, None
if current_mode == "Offline":
    s_eng, n_tok, n_mod = load_offline_engines()

# Main UI
st.title("🌎 Global Hybrid Translator")

col_l, col_r = st.columns(2)
with col_l: src_l = st.selectbox("From", list(ALL_LANGS.keys()), index=12) # Default English
with col_r: tgt_l = st.selectbox("To", list(ALL_LANGS.keys()), index=0)  # Default Hindi

st.divider()

# TAB 1: Manual Entry
tab_text, tab_voice = st.tabs(["📝 Text Input", "🎙️ Live Voice"])

with tab_text:
    t1, t2 = st.columns(2)
    with t1:
        manual_in = st.text_area("Type text manually:", height=150)
        if st.button("Translate Text ➔"):
            st.session_state.manual_out = perform_translation(manual_in, ALL_LANGS[src_l], ALL_LANGS[tgt_l], current_mode, n_tok, n_mod)
    with t2:
        st.text_area("Result:", value=st.session_state.manual_out, height=150, disabled=True)
        if st.button("🔊 Read Translation"):
            perform_tts(st.session_state.manual_out, ALL_LANGS[tgt_l], current_mode)

# TAB 2: Live Conversation
with tab_voice:
    st.write("Tap a mic and start speaking. The app will translate and read aloud automatically.")
    
    def handle_voice(audio_data, s_name, t_name):
        with open("live.wav", "wb") as f: f.write(audio_data['bytes'])
        if current_mode == "Online":
            r = sr.Recognizer()
            with sr.AudioFile("live.wav") as source:
                try: text = r.recognize_google(r.record(source))
                except: text = ""
        else:
            segments, _ = s_eng.transcribe("live.wav")
            text = " ".join([s.text for s in segments])

        if text.strip():
            translated = perform_translation(text, ALL_LANGS[s_name], ALL_LANGS[t_name], current_mode, n_tok, n_mod)
            st.chat_message("user").write(f"**{s_name}:** {text}")
            st.chat_message("assistant").write(f"**{t_name}:** {translated}")
            perform_tts(translated, ALL_LANGS[t_name], current_mode)
            # Log to DB
            conn = sqlite3.connect('global_translator.db')
            conn.execute("INSERT INTO logs VALUES (?,?,?,?,?,?)", ("User", text, translated, f"{s_name}->{t_name}", current_mode, datetime.now()))
            conn.commit()

    v1, v2 = st.columns(2)
    with v1:
        st.write(f"Mic: **{src_l}**")
        audio_1 = mic_recorder(start_prompt="Record", stop_prompt="Stop", key='m1')
        if audio_1: handle_voice(audio_1, src_l, tgt_l)
    with v2:
        st.write(f"Mic: **{tgt_l}**")
        audio_2 = mic_recorder(start_prompt="Record", stop_prompt="Stop", key='m2')
        if audio_2: handle_voice(audio_2, tgt_l, src_l)