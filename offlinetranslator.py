import streamlit as st
import sqlite3
import os
import hashlib
import io
import base64
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# 0. PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Global Hybrid Translator", layout="wide", page_icon="🌎")

# ─────────────────────────────────────────────────────────────
# 1. DATABASE
# ─────────────────────────────────────────────────────────────
DB = "global_translator.db"

def get_db():
    conn = sqlite3.connect(DB, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT UNIQUE NOT NULL,
            email     TEXT UNIQUE NOT NULL,
            pw_hash   TEXT NOT NULL,
            created   TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            user     TEXT,
            src_txt  TEXT,
            res_txt  TEXT,
            pair     TEXT,
            mode     TEXT,
            source   TEXT,
            time     TEXT
        )
    """)
    # Migration: add email column if it doesn't exist yet (for existing DBs)
    try:
        conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
        conn.commit()
    except Exception:
        pass
    # Migration: add source column to logs if missing
    try:
        conn.execute("ALTER TABLE logs ADD COLUMN source TEXT")
        conn.commit()
    except Exception:
        pass
    conn.commit()
    return conn

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def register_user(username: str, email: str, pw: str):
    conn = get_db()
    # Check email uniqueness manually for better error messages
    if conn.execute("SELECT 1 FROM users WHERE email=?", (email.strip().lower(),)).fetchone():
        return False, "Email already registered."
    try:
        conn.execute(
            "INSERT INTO users (username, email, pw_hash, created) VALUES (?,?,?,?)",
            (username.strip(), email.strip().lower(), hash_pw(pw), datetime.now().isoformat())
        )
        conn.commit()
        return True, "Account created!"
    except sqlite3.IntegrityError:
        return False, "Username already taken."

def login_user(identifier: str, pw: str):
    """Login with email OR username."""
    conn = get_db()
    ident = identifier.strip()
    row = conn.execute(
        "SELECT username, pw_hash FROM users WHERE email=? OR username=?",
        (ident.lower(), ident)
    ).fetchone()
    if row and row[1] == hash_pw(pw):
        return True, row[0]   # (success, username)
    return False, ""

def log_translation(user, src_txt, res_txt, pair, mode, source="text"):
    conn = get_db()
    conn.execute(
        "INSERT INTO logs (user, src_txt, res_txt, pair, mode, source, time) VALUES (?,?,?,?,?,?,?)",
        (user, src_txt, res_txt, pair, mode, source, datetime.now().isoformat())
    )
    conn.commit()

def get_user_history(user, limit=20):
    conn = get_db()
    return conn.execute(
        "SELECT src_txt, res_txt, pair, mode, source, time FROM logs WHERE user=? ORDER BY time DESC LIMIT ?",
        (user, limit)
    ).fetchall()

# ─────────────────────────────────────────────────────────────
# 2. LAZY OFFLINE ENGINE LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_offline_engines():
    from faster_whisper import WhisperModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    stt = WhisperModel("base", device="cpu", compute_type="int8")
    model_id = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return stt, tokenizer, model

# ─────────────────────────────────────────────────────────────
# 3. ESPEAK PATCH
# ─────────────────────────────────────────────────────────────
def _patch_espeak():
    try:
        from pyttsx3.drivers import espeak
        if not hasattr(espeak.EspeakDriver, "_is_patched"):
            _orig = espeak.EspeakDriver.setProperty
            def _patched(*args, **kwargs):
                try:
                    try: _orig(*args, **kwargs)
                    except TypeError as te:
                        if "positional argument" in str(te) and len(args) > 1:
                            _orig(*args[1:], **kwargs)
                        else: raise
                except ValueError as e:
                    if "SetVoiceByName failed" not in str(e): raise
            espeak.EspeakDriver.setProperty = _patched
            espeak.EspeakDriver._is_patched = True
    except Exception:
        pass
_patch_espeak()

# ─────────────────────────────────────────────────────────────
# 4. TTS HELPER  — returns audio bytes (no st.audio here)
# ─────────────────────────────────────────────────────────────
def generate_tts_bytes(text: str, lang_code: str, mode: str) -> tuple[bytes, str]:
    """Returns (audio_bytes, mime_type) or (b"", "")."""
    if not text.strip():
        return b"", ""
    online_code = lang_code.split("_")[0][:2]
    if mode == "Online":
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang=online_code).write_to_fp(buf)
        return buf.getvalue(), "audio/mp3"
    else:
        try:
            import pyttsx3, tempfile
            engine = pyttsx3.init()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                fname = f.name
            engine.save_to_file(text, fname)
            engine.runAndWait()
            with open(fname, "rb") as f:
                data = f.read()
            os.unlink(fname)
            return data, "audio/wav"
        except Exception as e:
            st.error(f"Offline TTS failed. Ensure 'espeak' is installed. Details: {e}")
            return b"", ""

def play_audio(audio_bytes: bytes, mime: str):
    if audio_bytes:
        st.audio(audio_bytes, format=mime, autoplay=True)

# ─────────────────────────────────────────────────────────────
# 5. TRANSLATION HELPER
# ─────────────────────────────────────────────────────────────
def perform_translation(text, src_code, tgt_code, mode, nllb_tok=None, nllb_model=None):
    if not text.strip():
        return ""
    if mode == "Online":
        from deep_translator import GoogleTranslator
        online_code = tgt_code.split("_")[0][:2]
        return GoogleTranslator(source="auto", target=online_code).translate(text)
    else:
        nllb_tok.src_lang = src_code
        inputs = nllb_tok(text, return_tensors="pt")
        tgt_id = nllb_tok.convert_tokens_to_ids(tgt_code)
        tokens = nllb_model.generate(**inputs, forced_bos_token_id=tgt_id, max_length=400)
        return nllb_tok.batch_decode(tokens, skip_special_tokens=True)[0]

# ─────────────────────────────────────────────────────────────
# 6. OCR — works fully offline via pytesseract
# ─────────────────────────────────────────────────────────────
def ocr_image(img_bytes: bytes) -> str:
    """Extract text from image bytes using Tesseract (offline)."""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes))
        return pytesseract.image_to_string(img).strip()
    except Exception as e:
        st.error(f"OCR failed. Ensure 'tesseract-ocr' is installed on your system. Details: {e}")
        return ""

def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        import fitz  # pymupdf
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        texts = []
        for page in doc:
            t = page.get_text()
            if t.strip():
                texts.append(t)
            else:
                # Scanned PDF page — rasterize and OCR
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("png")
                texts.append(ocr_image(img_bytes))
        return "\n".join(texts).strip()
    except Exception as e:
        st.error(f"PDF extraction failed: {e}")
        return ""

def extract_docx_text(docx_bytes: bytes) -> str:
    try:
        import docx as docx_lib
        doc = docx_lib.Document(io.BytesIO(docx_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        st.error(f"DOCX extraction failed: {e}")
        return ""

def extract_text_from_upload(file) -> str:
    name = file.name.lower()
    data = file.read()
    if name.endswith(".pdf"):
        return extract_pdf_text(data)
    elif name.endswith(".docx"):
        return extract_docx_text(data)
    elif name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    else:
        # Treat as image
        return ocr_image(data)

# ─────────────────────────────────────────────────────────────
# 7. LANGUAGE MAP
# ─────────────────────────────────────────────────────────────
ALL_LANGS = {
    "Hindi": "hin_Deva", "Bengali": "ben_Beng", "Tamil": "tam_Taml", "Telugu": "tel_Telu",
    "Marathi": "mar_Deva", "Gujarati": "guj_Gujr", "Kannada": "kan_Knda", "Malayalam": "mal_Mlym",
    "Punjabi": "pan_Guru", "Urdu": "urd_Arab", "Assamese": "asm_Beng", "Odia": "ory_Orya",
    "English": "eng_Latn", "Spanish": "spa_Latn", "French": "fra_Latn", "German": "deu_Latn",
    "Chinese (Simplified)": "zho_Hans", "Japanese": "jpn_Jpan", "Korean": "kor_Hang",
    "Russian": "rus_Cyrl", "Arabic": "arb_Arab", "Portuguese": "por_Latn", "Italian": "ita_Latn",
    "Turkish": "tur_Latn", "Vietnamese": "vie_Latn", "Thai": "tha_Thai",
}
LANG_KEYS = list(ALL_LANGS.keys())

# ─────────────────────────────────────────────────────────────
# 8. SESSION STATE
# ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "logged_in": False,
    "username": "",
    "manual_out": "",
    "manual_in_saved": "",   # so audio button can replay source
    "img_out": "",
    "img_in_text": "",
    "doc_out": "",
    "doc_in_text": "",
    "cam_out": "",
    "cam_in_text": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────
# 9. AUTH SCREEN
# ─────────────────────────────────────────────────────────────
def auth_screen():
    st.markdown("""
    <div style='text-align:center;padding:2.5rem 0 1.5rem'>
        <h1 style='font-size:2.8rem;margin-bottom:.3rem'>🌎 Global Hybrid Translator</h1>
        <p style='color:#888;font-size:1.05rem'>Translate text, voice, images & documents — online or offline</p>
    </div>""", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        tab_login, tab_reg = st.tabs(["🔐 Sign In", "📝 Create Account"])

        with tab_login:
            ident = st.text_input("Email or Username", key="li_ident",
                                  placeholder="you@email.com  or  username")
            pw    = st.text_input("Password", type="password", key="li_pw")
            if st.button("Sign In", use_container_width=True, type="primary"):
                ok, uname = login_user(ident, pw)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username  = uname
                    st.rerun()
                else:
                    st.error("Invalid credentials. Try again.")
            st.caption("🔒 Passwords stored as SHA-256 hashes only.")

        with tab_reg:
            nu   = st.text_input("Username",  key="reg_user", placeholder="e.g. arjun99")
            ne   = st.text_input("Email",     key="reg_email", placeholder="you@email.com")
            np   = st.text_input("Password",  type="password", key="reg_pw")
            np2  = st.text_input("Confirm Password", type="password", key="reg_pw2")
            if st.button("Create Account", use_container_width=True, type="primary"):
                if not nu or not ne or not np:
                    st.warning("All fields are required.")
                elif "@" not in ne:
                    st.warning("Enter a valid email address.")
                elif np != np2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register_user(nu, ne, np)
                    st.success(msg + " Please sign in.") if ok else st.error(msg)

# ─────────────────────────────────────────────────────────────
# 10. MAIN APP
# ─────────────────────────────────────────────────────────────
def main_app():
    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        if st.button("Sign Out", use_container_width=True):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

        st.divider()
        offline_on   = st.toggle("🔌 Offline Mode", value=False,
                                 help="Uses local AI — works without internet. First load downloads models.")
        current_mode = "Offline" if offline_on else "Online"
        badge_color  = "#e07b00" if current_mode == "Offline" else "#1a7f37"
        st.markdown(f"<span style='background:{badge_color};color:white;padding:3px 10px;"
                    f"border-radius:12px;font-size:.85rem'>● {current_mode}</span>",
                    unsafe_allow_html=True)

        s_eng = n_tok = n_mod = None
        if current_mode == "Offline":
            with st.spinner("Loading offline AI engines (one-time download)…"):
                s_eng, n_tok, n_mod = load_offline_engines()

        st.divider()
        st.markdown("### 📜 Your History")
        history = get_user_history(st.session_state.username, 15)
        if history:
            for src, res, pair, mode, source, ts in history:
                icon = {"text":"📝","image":"🖼️","document":"📄","camera":"📷","voice":"🎙️"}.get(source,"📝")
                with st.expander(f"{icon} {pair} · {ts[:16]}", expanded=False):
                    st.caption(f"Mode: {mode}  |  Source: {source}")
                    st.write(f"**In:** {src[:200]}")
                    st.write(f"**Out:** {res[:200]}")
        else:
            st.caption("No translations yet.")

    # ── Language pickers ──────────────────────────────────────
    st.title("🌎 Global Hybrid Translator")
    c1, c2 = st.columns(2)
    with c1: src_l = st.selectbox("From", LANG_KEYS, index=LANG_KEYS.index("English"))
    with c2: tgt_l = st.selectbox("To",   LANG_KEYS, index=LANG_KEYS.index("Hindi"))
    st.divider()

    # ── Tabs ─────────────────────────────────────────────────
    tab_text, tab_voice, tab_image, tab_doc, tab_cam = st.tabs([
        "📝 Text", "🎙️ Voice", "🖼️ Image OCR", "📄 Document", "📷 Camera"
    ])

    # ═══════════════════════════════════════════════════════════
    # TAB 1 — TEXT
    # ═══════════════════════════════════════════════════════════
    with tab_text:
        c1, c2 = st.columns(2)
        with c1:
            manual_in = st.text_area("Enter text:", height=170, placeholder="Type here…", key="txt_in")
            if st.button("Translate ➔", type="primary", use_container_width=True, key="btn_translate"):
                with st.spinner("Translating…"):
                    result = perform_translation(manual_in, ALL_LANGS[src_l], ALL_LANGS[tgt_l],
                                                 current_mode, n_tok, n_mod)
                    st.session_state.manual_out      = result
                    st.session_state.manual_in_saved = manual_in
                    if result:
                        log_translation(st.session_state.username, manual_in, result,
                                        f"{src_l}→{tgt_l}", current_mode, "text")
        with c2:
            st.text_area("Translation:", value=st.session_state.manual_out,
                         height=170, disabled=True, key="txt_out")

            ba1, ba2, ba3 = st.columns(3)
            with ba1:
                if st.button("🔊 Read Input", use_container_width=True, key="btn_read_in"):
                    ab, mime = generate_tts_bytes(st.session_state.manual_in_saved,
                                                  ALL_LANGS[src_l], current_mode)
                    play_audio(ab, mime)
            with ba2:
                if st.button("🔊 Read Output", use_container_width=True, key="btn_read_out"):
                    ab, mime = generate_tts_bytes(st.session_state.manual_out,
                                                  ALL_LANGS[tgt_l], current_mode)
                    play_audio(ab, mime)
            with ba3:
                if st.button("🔊 Read Both", use_container_width=True, key="btn_read_both"):
                    with st.spinner("Generating audio…"):
                        ab1, m1 = generate_tts_bytes(st.session_state.manual_in_saved,
                                                     ALL_LANGS[src_l], current_mode)
                        ab2, m2 = generate_tts_bytes(st.session_state.manual_out,
                                                     ALL_LANGS[tgt_l], current_mode)
                    if ab1:
                        st.caption(f"▶ {src_l}")
                        play_audio(ab1, m1)
                    if ab2:
                        st.caption(f"▶ {tgt_l}")
                        play_audio(ab2, m2)

    # ═══════════════════════════════════════════════════════════
    # TAB 2 — VOICE
    # ═══════════════════════════════════════════════════════════
    with tab_voice:
        st.info("Tap **Record**, speak, then **Stop**. App transcribes → translates → reads aloud.")

        def handle_voice(audio_data, speaker_lang, listener_lang):
            with open("live.wav", "wb") as f:
                f.write(audio_data["bytes"])
            if current_mode == "Online":
                import speech_recognition as sr
                r = sr.Recognizer()
                with sr.AudioFile("live.wav") as src:
                    try: text = r.recognize_google(r.record(src))
                    except: text = ""
            else:
                segs, _ = s_eng.transcribe("live.wav")
                text = " ".join([s.text for s in segs])

            if text.strip():
                translated = perform_translation(text, ALL_LANGS[speaker_lang],
                                                 ALL_LANGS[listener_lang],
                                                 current_mode, n_tok, n_mod)
                st.chat_message("user").write(f"**{speaker_lang}:** {text}")
                st.chat_message("assistant").write(f"**{listener_lang}:** {translated}")
                ab, mime = generate_tts_bytes(translated, ALL_LANGS[listener_lang], current_mode)
                play_audio(ab, mime)
                log_translation(st.session_state.username, text, translated,
                                f"{speaker_lang}→{listener_lang}", current_mode, "voice")
            else:
                st.warning("No speech detected. Try again.")

        v1, v2 = st.columns(2)
        with v1:
            st.markdown(f"**🎤 {src_l}**")
            from streamlit_mic_recorder import mic_recorder
            a1 = mic_recorder(start_prompt="▶ Record", stop_prompt="⏹ Stop", key="m1")
            if a1: handle_voice(a1, src_l, tgt_l)
        with v2:
            st.markdown(f"**🎤 {tgt_l}**")
            a2 = mic_recorder(start_prompt="▶ Record", stop_prompt="⏹ Stop", key="m2")
            if a2: handle_voice(a2, tgt_l, src_l)

    # ═══════════════════════════════════════════════════════════
    # TAB 3 — IMAGE OCR
    # ═══════════════════════════════════════════════════════════
    with tab_image:
        st.markdown("Upload any image containing text — the app extracts and translates it **fully offline** using Tesseract OCR.")
        uploaded_img = st.file_uploader("Upload image (PNG / JPG / JPEG / WEBP / BMP)",
                                        type=["png","jpg","jpeg","webp","bmp"], key="img_up")
        if uploaded_img:
            st.image(uploaded_img, caption="Uploaded image", use_column_width=True)
            if st.button("Extract & Translate Image Text ➔", type="primary",
                         use_container_width=True, key="btn_img"):
                with st.spinner("Running OCR…"):
                    raw = ocr_image(uploaded_img.read())
                if raw:
                    st.session_state.img_in_text = raw
                    with st.spinner("Translating…"):
                        result = perform_translation(raw, ALL_LANGS[src_l], ALL_LANGS[tgt_l],
                                                     current_mode, n_tok, n_mod)
                    st.session_state.img_out = result
                    log_translation(st.session_state.username, raw, result,
                                    f"{src_l}→{tgt_l}", current_mode, "image")
                else:
                    st.warning("No text found in image. Ensure text is clear and Tesseract is installed.")

        if st.session_state.img_in_text or st.session_state.img_out:
            ci1, ci2 = st.columns(2)
            with ci1:
                st.markdown("**📋 Extracted Text**")
                st.text_area("", value=st.session_state.img_in_text, height=180,
                             disabled=True, key="img_extracted")
                if st.button("🔊 Read Extracted", key="img_audio_in"):
                    ab, mime = generate_tts_bytes(st.session_state.img_in_text,
                                                  ALL_LANGS[src_l], current_mode)
                    play_audio(ab, mime)
            with ci2:
                st.markdown("**🌐 Translation**")
                st.text_area("", value=st.session_state.img_out, height=180,
                             disabled=True, key="img_translated")
                if st.button("🔊 Read Translation", key="img_audio_out"):
                    ab, mime = generate_tts_bytes(st.session_state.img_out,
                                                  ALL_LANGS[tgt_l], current_mode)
                    play_audio(ab, mime)

    # ═══════════════════════════════════════════════════════════
    # TAB 4 — DOCUMENT
    # ═══════════════════════════════════════════════════════════
    with tab_doc:
        st.markdown("Upload a **PDF**, **DOCX**, or **TXT** file. Scanned PDFs are OCR-processed automatically — works offline.")
        uploaded_doc = st.file_uploader("Upload document",
                                        type=["pdf","docx","txt"], key="doc_up")
        if uploaded_doc:
            col_info, col_btn = st.columns([3, 1])
            with col_info:
                st.success(f"📎 {uploaded_doc.name}  ({uploaded_doc.size // 1024} KB)")
            with col_btn:
                run_doc = st.button("Extract & Translate ➔", type="primary",
                                    use_container_width=True, key="btn_doc")
            if run_doc:
                with st.spinner("Extracting text from document…"):
                    raw = extract_text_from_upload(uploaded_doc)
                if raw:
                    st.session_state.doc_in_text = raw
                    # Chunk long documents for NLLB (max ~512 tokens per call)
                    with st.spinner("Translating document…"):
                        if current_mode == "Offline" and len(raw) > 1500:
                            # Split by sentences/paragraphs to stay within model limit
                            import re
                            chunks = re.split(r'(?<=[.!?।])\s+|\n{2,}', raw)
                            translated_chunks = []
                            progress = st.progress(0)
                            for i, chunk in enumerate(chunks):
                                if chunk.strip():
                                    translated_chunks.append(
                                        perform_translation(chunk, ALL_LANGS[src_l],
                                                            ALL_LANGS[tgt_l], current_mode,
                                                            n_tok, n_mod)
                                    )
                                progress.progress((i + 1) / len(chunks))
                            result = " ".join(translated_chunks)
                            progress.empty()
                        else:
                            result = perform_translation(raw, ALL_LANGS[src_l], ALL_LANGS[tgt_l],
                                                         current_mode, n_tok, n_mod)
                    st.session_state.doc_out = result
                    log_translation(st.session_state.username,
                                    raw[:500] + ("…" if len(raw) > 500 else ""),
                                    result[:500] + ("…" if len(result) > 500 else ""),
                                    f"{src_l}→{tgt_l}", current_mode, "document")
                else:
                    st.warning("Could not extract text from this document.")

        if st.session_state.doc_in_text or st.session_state.doc_out:
            cd1, cd2 = st.columns(2)
            with cd1:
                st.markdown("**📋 Extracted Text**")
                st.text_area("", value=st.session_state.doc_in_text, height=260,
                             disabled=True, key="doc_extracted")
                if st.button("🔊 Read Extracted", key="doc_audio_in"):
                    # TTS first 1000 chars (very long docs would be impractical)
                    ab, mime = generate_tts_bytes(st.session_state.doc_in_text[:1000],
                                                  ALL_LANGS[src_l], current_mode)
                    play_audio(ab, mime)
            with cd2:
                st.markdown("**🌐 Translation**")
                st.text_area("", value=st.session_state.doc_out, height=260,
                             disabled=True, key="doc_translated")
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("🔊 Read Translation", key="doc_audio_out"):
                        ab, mime = generate_tts_bytes(st.session_state.doc_out[:1000],
                                                      ALL_LANGS[tgt_l], current_mode)
                        play_audio(ab, mime)
                with b2:
                    # Download translated text
                    if st.session_state.doc_out:
                        st.download_button("⬇ Download", data=st.session_state.doc_out,
                                           file_name="translation.txt", mime="text/plain",
                                           use_container_width=True, key="doc_dl")

    # ═══════════════════════════════════════════════════════════
    # TAB 5 — CAMERA
    # ═══════════════════════════════════════════════════════════
    with tab_cam:
        st.markdown("Take a photo with your device camera. Text in the photo is extracted using OCR and translated — **works offline**.")
        cam_img = st.camera_input("📷 Point camera at text and capture", key="cam_input")
        if cam_img:
            if st.button("Extract & Translate from Camera ➔", type="primary",
                         use_container_width=True, key="btn_cam"):
                with st.spinner("Running OCR on captured image…"):
                    raw = ocr_image(cam_img.getvalue())
                if raw:
                    st.session_state.cam_in_text = raw
                    with st.spinner("Translating…"):
                        result = perform_translation(raw, ALL_LANGS[src_l], ALL_LANGS[tgt_l],
                                                     current_mode, n_tok, n_mod)
                    st.session_state.cam_out = result
                    log_translation(st.session_state.username, raw, result,
                                    f"{src_l}→{tgt_l}", current_mode, "camera")
                else:
                    st.warning("No text detected. Ensure text is clearly visible and well-lit. "
                               "Also verify Tesseract OCR is installed on the server.")

        if st.session_state.cam_in_text or st.session_state.cam_out:
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**📋 Detected Text**")
                st.text_area("", value=st.session_state.cam_in_text, height=160,
                             disabled=True, key="cam_extracted")
                if st.button("🔊 Read Detected", key="cam_audio_in"):
                    ab, mime = generate_tts_bytes(st.session_state.cam_in_text,
                                                  ALL_LANGS[src_l], current_mode)
                    play_audio(ab, mime)
            with cc2:
                st.markdown("**🌐 Translation**")
                st.text_area("", value=st.session_state.cam_out, height=160,
                             disabled=True, key="cam_translated")
                if st.button("🔊 Read Translation", key="cam_audio_out"):
                    ab, mime = generate_tts_bytes(st.session_state.cam_out,
                                                  ALL_LANGS[tgt_l], current_mode)
                    play_audio(ab, mime)

# ─────────────────────────────────────────────────────────────
# 11. ROUTER
# ─────────────────────────────────────────────────────────────
if st.session_state.logged_in:
    main_app()
else:
    auth_screen()
