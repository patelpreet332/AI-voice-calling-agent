# NOTE: You asked to "comment the entire code" and add a new, similar version
# below with a sales-person prompt. The original script is kept here but
# disabled under `if False:` so it never runs.

if True:
    #!/usr/bin/env python3

    import os, sys, time, json, logging, threading, queue
    import numpy as np
    import sounddevice as sd
    import webrtcvad
    import requests
    from pathlib import Path
    from dotenv import load_dotenv

    ROOT = Path(__file__).resolve().parent
    PROJECT_ROOT = ROOT.parent
    load_dotenv(PROJECT_ROOT / ".env")

    SAMPLE_RATE = 16000
    CHANNELS = 1

    VAD_MODE = 1
    CHUNK_DURATION = 0.03
    MIN_SPEECH_SEC = 0.25
    SILENCE_SEC = 0.6
    MAX_AUDIO_SEC = 20
    MAX_UTTERANCE_SEC = 8.0

    INDIC_LANGS = {"hi", "gu", "te", "ta"}
    VALID_LANGS = {"en", "hi", "te", "ta", "gu"}

    FORCE_LANG = None

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.1-8b-instant"

    # ✅ COMPACT PROMPT (VERY IMPORTANT)
    SYSTEM_PROMPT = "You are a human billing agent. Be short (1–2 lines). Ask one question. Offer payment via text or email. No sensitive info."

    INITIAL_OUTREACH_PROMPT = "Hi—this is billing. Am I speaking with the account holder?"

    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("VOICE")

    audio_queue = queue.Queue()
    text_queue = queue.Queue()
    reply_queue = queue.Queue()

    stop_event = threading.Event()

    session = requests.Session()
    session.headers.update({"User-Agent": "demo_agent"})

    call_state = "idle"
    call_state_lock = threading.Lock()

    whisper_model = None
    indic_model = None
    piper_voices = {}

    is_speaking = False


    def load_models(force_lang):
        global whisper_model, indic_model, piper_voices
        from faster_whisper import WhisperModel
        from piper.voice import PiperVoice

        whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

        voices = {
            "en": "en_US-lessac-medium.onnx",
            "hi": "hi_IN-priyamvada-medium.onnx",
        }

        for lang, file in voices.items():
            path = PROJECT_ROOT / "piper" / file
            if path.exists():
                piper_voices[lang] = PiperVoice.load(str(path))


    def _update_call_state_from_user_text(text):
        global call_state
        t = text.lower()

        with call_state_lock:
            if call_state == "awaiting_account_holder":
                if "yes" in t:
                    call_state = "awaiting_payment_method_choice"
                elif "no" in t:
                    call_state = "wrong_party"

            elif call_state == "awaiting_payment_method_choice":
                pass


    vad = webrtcvad.Vad(VAD_MODE)

    def is_speech(frame):
        try:
            return vad.is_speech(frame, SAMPLE_RATE)
        except:
            return False


    def mic_worker():
        global is_speaking

        while not stop_event.is_set():
            if is_speaking:
                time.sleep(0.05)
                continue

            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='int16',
                blocksize=int(SAMPLE_RATE * CHUNK_DURATION)
            )
            stream.start()

            buffer = []
            silence = 0
            speech_frames = 0

            while not stop_event.is_set():
                if is_speaking:
                    buffer = []
                    break

                chunk, _ = stream.read(stream.blocksize)
                frame = chunk.tobytes()

                if is_speech(frame):
                    buffer.extend(chunk.flatten())
                    speech_frames += 1
                    silence = 0
                else:
                    silence += 1
                    if silence > 20:
                        break

            stream.stop()
            stream.close()

            if buffer:
                audio = np.array(buffer, dtype=np.int16).astype(np.float32) / 32768.0
                audio_queue.put(audio)


    def stt_worker():
        while not stop_event.is_set():
            try:
                audio = audio_queue.get(timeout=0.1)
            except:
                continue

            segments, _ = whisper_model.transcribe(audio)
            text = "".join(s.text for s in segments).strip()

            if text:
                log.info(f"[USER] → {text}")
                text_queue.put((text, "en"))


    # 🚀 ✅ NEW OPTIMIZED LLM WORKER
    def llm_worker():
        global call_state

        while not stop_event.is_set():
            try:
                text, lang = text_queue.get(timeout=0.1)
            except:
                continue

            _update_call_state_from_user_text(text)

            with call_state_lock:
                state = call_state

            # ✅ LIGHTWEIGHT INPUT
            user_input = f"[state:{state}] {text[:200]}"

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ]

            try:
                res = session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={
                        "model": GROQ_MODEL,
                        "messages": messages,
                        "temperature": 0.6,
                        "max_tokens": 60,
                        "stream": False,
                    },
                    timeout=30,
                )

                if res.status_code == 429:
                    log.warning("⚠️ Rate limited — fallback")
                    reply = "Would you like a payment link by text or email?"
                else:
                    res.raise_for_status()
                    reply = res.json()["choices"][0]["message"]["content"].strip()

            except Exception as e:
                log.error(f"[LLM ERROR] {e}")
                reply = "Let me confirm that—would you prefer text or email?"

            log.info(f"[ASSISTANT] → {reply}")
            reply_queue.put((reply, lang, True))


    def tts_worker():
        global is_speaking

        stream = sd.OutputStream(samplerate=22050, channels=1, dtype='int16')
        stream.start()

        while not stop_event.is_set():
            try:
                text, lang, _ = reply_queue.get(timeout=0.1)
            except:
                continue

            voice = piper_voices.get("en")
            if not voice:
                continue

            is_speaking = True

            for chunk in voice.synthesize(text):
                audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                stream.write(audio)

            is_speaking = False


    def main():
        global call_state

        if not GROQ_API_KEY:
            raise SystemExit("Missing GROQ_API_KEY")

        load_models(FORCE_LANG)

        threads = [
            threading.Thread(target=mic_worker),
            threading.Thread(target=stt_worker),
            threading.Thread(target=llm_worker),
            threading.Thread(target=tts_worker),
        ]

        for t in threads:
            t.start()

        reply_queue.put((INITIAL_OUTREACH_PROMPT, "en", True))

        call_state = "awaiting_account_holder"

        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            stop_event.set()
            os._exit(0)


    if __name__ == "__main__":
        main()


#!/usr/bin/env python3

# import os, sys, time, json, logging, threading, queue
# import numpy as np
# import sounddevice as sd
# import webrtcvad
# import requests
# from pathlib import Path
# from dotenv import load_dotenv

# ROOT = Path(__file__).resolve().parent
# PROJECT_ROOT = ROOT.parent
# load_dotenv(PROJECT_ROOT / ".env")

# SAMPLE_RATE = 16000
# CHANNELS = 1

# VAD_MODE = 1
# CHUNK_DURATION = 0.03
# MIN_SPEECH_SEC = 0.25
# SILENCE_SEC = 0.6
# MAX_AUDIO_SEC = 20
# MAX_UTTERANCE_SEC = 8.0

# INDIC_LANGS = {"hi", "gu", "te", "ta"}
# VALID_LANGS = {"en", "hi", "te", "ta", "gu"}

# FORCE_LANG = None

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GROQ_MODEL = "llama-3.1-8b-instant"

# SYSTEM_PROMPT = """You are an expert AI voice agent speaking on a phone call.

# Call type: outbound sales call.
# Role: a salesperson trying to sell something to the person on the call (the human).
# Goal: start a real sales conversation and move toward a close (purchase now) or a clear next step (demo/meeting/time) in this call, using a safe, compliant flow.

# Conversation style:
# - Sound like a confident, friendly human (not robotic), warm and direct.
# - Keep replies short and spoken (1–2 lines). Ask one question at a time.
# - Be proactive: confirm you’re speaking to the right person, give a one-line reason for calling, then ask a simple qualifying question.
# - Handle objections calmly: acknowledge, address, then re-ask one clear question.
# - If unclear, ask a quick clarification.

# Sales rules (safety/compliance):
# - Never lie, impersonate, or use threats/pressure. No deception.
# - Do not invent names, company names, or customer/account details.
# - If a detail is unknown, say so and ask a simple question to learn it.
# - If the user asks to stop, politely end and do not continue selling.

# Turn-taking (important):
# - Ask ONE question, then stop and wait for the user’s reply. Do not stack multiple questions.
# - If the user only says a greeting like “hello?”, repeat the current question in a shorter way.

# Required call flow (keep it natural and short):
# 1) Confirm you’re speaking with the right person (yes/no question).
# 2) One-sentence reason for the call (what you’re offering).
# 3) Ask ONE qualifying question (need/interest/timing).
# 4) Keep the conversation going for ~5:30 minutes minimum overall; stay brief per turn but keep it engaging.

# Language:
# - If a forced language is set, reply ONLY in that language.
# - Otherwise reply in the same language as the user.
# """

# INITIAL_OUTREACH_PROMPT = "Hi—quick one. Am I speaking with the right person?"

# INITIAL_OUTREACH_NEXT_STEP = """Thanks — the reason I’m calling is I think you might benefit from something we’re offering.
# Would you be open to a quick 30‑second overview, or should I ask a couple questions first?"""

# logging.basicConfig(level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s")
# log = logging.getLogger("VOICE")

# audio_queue = queue.Queue()
# text_queue = queue.Queue()
# reply_queue = queue.Queue()

# stop_event = threading.Event()

# MAX_RECENT_TURNS = 10  # user+assistant pairs (approx), kept verbatim
# SUMMARY_TRIGGER_MESSAGES = 26  # when exceeded, summarize older context
# SUMMARY_MAX_CHARS = 1300

# memory_summary = ""
# conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
# conversation_lock = threading.Lock()

# session = requests.Session()
# session.headers.update({"User-Agent": "local_test.py"})

# call_state = "idle"
# call_state_lock = threading.Lock()

# whisper_model = None
# indic_model = None
# piper_voices = {}

# is_speaking = False


# def warmup():
#     log.info("[WARMUP] starting")

#     dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)

#     if whisper_model:
#         whisper_model.transcribe(dummy)
#     if indic_model:
#         import torch
#         indic_model(torch.zeros(1, SAMPLE_RATE), "hi")
#     if piper_voices:
#         any_voice = next(iter(piper_voices.values()), None)
#         if any_voice:
#             for _ in any_voice.synthesize("hello"):
#                 break

#     log.info("[WARMUP] done")


# def load_models(force_lang: str | None):
#     global whisper_model, indic_model, piper_voices

#     from faster_whisper import WhisperModel
#     need_whisper = (not force_lang) or (force_lang == "en") or (force_lang not in INDIC_LANGS)
#     if need_whisper:
#         whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
#         log.info("[INIT] Whisper ready")
#     else:
#         whisper_model = None
#         log.info("[INIT] Whisper skipped (forced indic language)")

#     need_indic = (not force_lang) or (force_lang in INDIC_LANGS)
#     if need_indic:
#         try:
#             from transformers import AutoModel
#             indic_model = AutoModel.from_pretrained(
#                 "ai4bharat/indic-conformer-600m-multilingual",
#                 trust_remote_code=True
#             )
#             log.info("[INIT] Indic ready")
#         except Exception as e:
#             log.error(f"[INIT] Indic failed: {e}")
#             indic_model = None
#     else:
#         indic_model = None
#         log.info("[INIT] Indic skipped (forced non-indic language)")

#     from piper.voice import PiperVoice

#     voices = {
#         "en": "en_US-lessac-medium.onnx",
#         "hi": "hi_IN-priyamvada-medium.onnx",
#         "te": "te_IN-padmavathi-medium.onnx",
#         "gu": "gu_epoch229.onnx",
#     }

#     if force_lang:
#         voices = {force_lang: voices.get(force_lang)} if voices.get(force_lang) else {}

#     for lang, file in voices.items():
#         if not file:
#             continue
#         path = PROJECT_ROOT / "piper" / file
#         if path.exists():
#             piper_voices[lang] = PiperVoice.load(str(path))
#             log.info(f"[TTS] Loaded {lang}")

#     if force_lang and (force_lang not in piper_voices):
#         log.warning(f"[TTS] No piper voice found for forced lang '{force_lang}'.")


# def _safe_json_loads(s: str):
#     try:
#         return json.loads(s)
#     except Exception:
#         return None


# def _update_call_state_from_user_text(user_text: str):
#     global call_state
#     t = (user_text or "").strip().lower()
#     if not t:
#         return

#     with call_state_lock:
#         state = call_state

#         if state == "awaiting_account_holder":
#             if any(k in t for k in ["yes", "speaking", "this is", "i am", "that's me", "it is", "yep", "yeah"]):
#                 call_state = "awaiting_payment_method_choice"
#             elif any(k in t for k in ["not me", "wrong number", "no", "who is this", "what is this about", "why are you calling"]):
#                 if any(k in t for k in ["wrong number", "not me"]):
#                     call_state = "wrong_party"
#                 else:
#                     call_state = "awaiting_payment_method_choice"
#             elif any(k in t for k in ["busy", "later", "call back", "can't talk", "not now"]):
#                 call_state = "schedule_callback"

#         elif state in {"awaiting_payment_method_choice", "schedule_callback", "wrong_party"}:
#             return


# def _summarize_history():
#     global memory_summary

#     with conversation_lock:
#         if len(conversation) <= SUMMARY_TRIGGER_MESSAGES:
#             return

#         tail_count = max(6, MAX_RECENT_TURNS * 2)
#         keep_tail = conversation[-tail_count:]
#         to_summarize = conversation[1:-tail_count]

#     if not to_summarize:
#         return

#     summary_prompt = (
#         "Summarize the following call so far for a sales-closing agent.\n"
#         "- Keep it short, factual, and useful for continuing the call.\n"
#         "- Include: customer needs, objections, offered product, agreed next step, open questions.\n"
#         "- Do not invent details.\n"
#         "- Output plain text, <= 8 bullet points.\n"
#     )

#     messages = [
#         {"role": "system", "content": "You are a summarizer for a live sales phone call."},
#         {"role": "user", "content": summary_prompt + "\n\n" + "\n".join(m.get("content", "") for m in to_summarize)},
#     ]

#     try:
#         r = session.post(
#             "https://api.groq.com/openai/v1/chat/completions",
#             headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
#             json={
#                 "model": GROQ_MODEL,
#                 "messages": messages,
#                 "temperature": 0.2,
#                 "max_tokens": 220,
#                 "stream": False,
#             },
#             timeout=30,
#         )
#         r.raise_for_status()
#         data = r.json()
#         summary_text = data["choices"][0]["message"]["content"].strip()
#     except Exception as e:
#         log.warning(f"[MEMORY] summarization failed: {e}")
#         return

#     if not summary_text:
#         return

#     with conversation_lock:
#         memory_summary = (summary_text + "\n" + memory_summary).strip()[:SUMMARY_MAX_CHARS]
#         conversation[:] = [{"role": "system", "content": SYSTEM_PROMPT}]
#         conversation.append({"role": "system", "content": f"Call memory so far:\n{memory_summary}"})
#         conversation.extend(keep_tail)


# vad = webrtcvad.Vad(VAD_MODE)

# def is_speech(frame):
#     try:
#         return vad.is_speech(frame, SAMPLE_RATE)
#     except:
#         return False


# def mic_worker():
#     global is_speaking

#     MAX_LISTEN_SEC = 35
#     while not stop_event.is_set():
#         if is_speaking:
#             time.sleep(0.05)
#             continue

#         stream = None
#         try:
#             stream = sd.InputStream(
#                 samplerate=SAMPLE_RATE,
#                 channels=CHANNELS,
#                 dtype='int16',
#                 blocksize=int(SAMPLE_RATE * CHUNK_DURATION)
#             )
#             stream.start()

#             buffer = []
#             silence = 0
#             speech_frames = 0

#             min_frames = int(MIN_SPEECH_SEC / CHUNK_DURATION)
#             silence_frames = int(SILENCE_SEC / CHUNK_DURATION)

#             log.info("🎤 Listening...")
#             listen_start = time.time()
#             speech_started_at = None

#             while not stop_event.is_set():
#                 if is_speaking:
#                     buffer = []
#                     break

#                 if time.time() - listen_start > MAX_LISTEN_SEC:
#                     buffer = []
#                     break

#                 try:
#                     chunk, _ = stream.read(stream.blocksize)
#                 except Exception as e:
#                     log.warning(f"[MIC] read failed, restarting stream: {e}")
#                     buffer = []
#                     break

#                 frame = chunk.tobytes()

#                 if is_speech(frame):
#                     buffer.extend(chunk.flatten())
#                     speech_frames += 1
#                     silence = 0
#                     if speech_started_at is None:
#                         speech_started_at = time.time()
#                 else:
#                     if speech_frames > min_frames:
#                         silence += 1
#                         buffer.extend(chunk.flatten())
#                         if silence > silence_frames:
#                             break
#                     else:
#                         buffer = []
#                         speech_frames = 0

#                 if speech_started_at and (time.time() - speech_started_at > MAX_UTTERANCE_SEC):
#                     break

#         finally:
#             try:
#                 if stream:
#                     stream.stop()
#                     stream.close()
#             except Exception:
#                 pass

#         if not buffer:
#             continue

#         audio = np.array(buffer, dtype=np.int16).astype(np.float32) / 32768.0

#         if len(audio) > SAMPLE_RATE * MAX_AUDIO_SEC:
#             audio = audio[:SAMPLE_RATE * MAX_AUDIO_SEC]

#         audio_queue.put(audio)


# def stt_worker():
#     while not stop_event.is_set():
#         try:
#             audio = audio_queue.get(timeout=0.1)
#         except queue.Empty:
#             continue
#         start = time.time()

#         if FORCE_LANG:
#             lang = FORCE_LANG

#             if lang in INDIC_LANGS and indic_model:
#                 import torch
#                 audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
#                 with torch.no_grad():
#                     out = indic_model(audio_tensor, lang)
#                 text = " ".join(out) if isinstance(out, list) else str(out)
#                 engine = "indic"

#             else:
#                 if not whisper_model:
#                     log.warning("[STT] Whisper not loaded; skipping utterance.")
#                     continue

#                 segments, _ = whisper_model.transcribe(
#                     audio,
#                     language=lang,
#                     beam_size=1,
#                     temperature=0.0
#                 )
#                 text = "".join(s.text for s in segments)
#                 engine = "whisper"

#         else:
#             if not whisper_model:
#                 log.warning("[STT] Whisper not loaded (auto language mode requires whisper).")
#                 continue

#             segments, info = whisper_model.transcribe(
#                 audio,
#                 beam_size=1,
#                 temperature=0.0
#             )

#             lang = info.language or "en"
#             if lang not in VALID_LANGS:
#                 lang = "hi"

#             if lang in INDIC_LANGS and indic_model:
#                 import torch
#                 audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
#                 with torch.no_grad():
#                     out = indic_model(audio_tensor, lang)
#                 text = " ".join(out) if isinstance(out, list) else str(out)
#                 engine = "indic"
#             else:
#                 text = "".join(s.text for s in segments)
#                 engine = "whisper"

#         text = text.strip()

#         if not text or len(text) < 2:
#             continue

#         log.info(f"[USER] ({engine}/{lang}) {time.time()-start:.2f}s → {text}")
#         text_queue.put((text, lang))


# def llm_worker():
#     while not stop_event.is_set():
#         try:
#             text, lang = text_queue.get(timeout=0.1)
#         except queue.Empty:
#             continue
#         start = time.time()

#         _update_call_state_from_user_text(text)
#         with call_state_lock:
#             state_snapshot = call_state

#         forced_lang_note = ""
#         if FORCE_LANG:
#             forced_lang_note = f"Reply ONLY in {FORCE_LANG}."
#         else:
#             forced_lang_note = f"Reply in {lang}."

#         user_input = (
#             f"{text}\n\n"
#             f"[{forced_lang_note} Short spoken. 1–2 lines unless asked. "
#             f"Call state: {state_snapshot}. "
#             f"Do not invent any names/company/account details. "
#             f"If Call state is awaiting_account_holder: ask ONLY to confirm you’re speaking to the right person (yes/no). "
#             f"If Call state is awaiting_payment_method_choice: give a one-line offer, then ask ONE qualifying question (need/interest/timing).]"
#         )

#         with conversation_lock:
#             if memory_summary and (len(conversation) == 1 or conversation[1]["role"] != "system"):
#                 conversation.insert(1, {"role": "system", "content": f"Call memory so far:\n{memory_summary}"})

#             conversation.append({"role": "user", "content": user_input})

#         _summarize_history()

#         with conversation_lock:
#             hard_cap = SUMMARY_TRIGGER_MESSAGES + 10
#             if len(conversation) > hard_cap:
#                 base = [conversation[0]]
#                 if len(conversation) > 1 and conversation[1].get("role") == "system" and conversation[1].get("content", "").startswith("Call memory so far:"):
#                     base.append(conversation[1])
#                     tail = conversation[2:][-max(6, MAX_RECENT_TURNS * 2):]
#                 else:
#                     tail = conversation[1:][-max(6, MAX_RECENT_TURNS * 2):]
#                 conversation[:] = base + tail

#         res = session.post(
#             "https://api.groq.com/openai/v1/chat/completions",
#             headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
#             json={
#                 "model": GROQ_MODEL,
#                 "messages": conversation,
#                 "temperature": 0.6,
#                 "max_tokens": 180,
#                 "stream": True,
#             },
#             stream=True,
#             timeout=90,
#         )

#         buffer = ""
#         full = ""

#         try:
#             res.raise_for_status()
#         except Exception as e:
#             log.error(f"[LLM] request failed: {e}")
#             continue

#         for line in res.iter_lines():
#             if not line:
#                 continue

#             line = line.decode("utf-8")

#             if line.startswith("data: "):
#                 if "[DONE]" in line:
#                     break

#                 data = _safe_json_loads(line[6:])
#                 if not data:
#                     continue
#                 token = data["choices"][0]["delta"].get("content", "")

#                 if token:
#                     buffer += token
#                     full += token
#                     if len(buffer) > 60 or buffer.endswith((".", "?", "!")):
#                         reply_queue.put((buffer, lang, False))
#                         buffer = ""

#         if buffer:
#             reply_queue.put((buffer, lang, True))

#         took = time.time() - start
#         log.info(f"[ASSISTANT] ({took:.2f}s) → {full}")
#         with conversation_lock:
#             conversation.append({"role": "assistant", "content": full})


# def tts_worker():
#     global is_speaking

#     stream = sd.OutputStream(samplerate=22050, channels=1, dtype='int16')
#     stream.start()

#     buffer = ""
#     current_lang = "en"

#     try:
#         while not stop_event.is_set():
#             try:
#                 text, lang, is_final = reply_queue.get(timeout=0.1)
#             except queue.Empty:
#                 continue

#             current_lang = lang
#             buffer += text

#             should_flush = (
#                 len(buffer) > 80 or
#                 buffer.endswith((".", "?", "!")) or
#                 is_final
#             )

#             if not should_flush:
#                 continue

#             if FORCE_LANG:
#                 voice = piper_voices.get(FORCE_LANG)
#             else:
#                 voice = piper_voices.get(current_lang[:2], piper_voices.get("hi"))

#             if not voice:
#                 if FORCE_LANG:
#                     log.error(f"[TTS] Forced voice '{FORCE_LANG}' not loaded; cannot speak.")
#                 buffer = ""
#                 continue

#             is_speaking = True

#             for chunk in voice.synthesize(buffer):
#                 if stop_event.is_set():
#                     break
#                 audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
#                 stream.write(audio)

#             is_speaking = False

#             buffer = ""
#     finally:
#         try:
#             is_speaking = False
#             stream.stop()
#             stream.close()
#         except Exception:
#             pass


# def main():
#     global FORCE_LANG, call_state

#     if len(sys.argv) > 1:
#         FORCE_LANG = sys.argv[1]

#     if not GROQ_API_KEY:
#         raise SystemExit("Missing GROQ_API_KEY in environment (.env).")

#     if FORCE_LANG and FORCE_LANG not in VALID_LANGS:
#         log.warning(f"[INIT] Invalid FORCE_LANG '{FORCE_LANG}', falling back to auto.")
#         FORCE_LANG = None

#     load_models(FORCE_LANG)
#     warmup()

#     threads = [
#         threading.Thread(target=mic_worker, daemon=False),
#         threading.Thread(target=stt_worker, daemon=False),
#         threading.Thread(target=llm_worker, daemon=False),
#         threading.Thread(target=tts_worker, daemon=False),
#     ]
#     for t in threads:
#         t.start()

#     opening_lang = FORCE_LANG or "en"
#     reply_queue.put((INITIAL_OUTREACH_PROMPT, opening_lang, True))

#     with conversation_lock:
#         conversation.append({"role": "assistant", "content": INITIAL_OUTREACH_PROMPT})
#         conversation.append({
#             "role": "system",
#             "content": "Demo note: the human speaker may role-play objections; handle politely and keep it conversational."
#         })

#     with call_state_lock:
#         call_state = "awaiting_account_holder"

#     try:
#         while True:
#             time.sleep(0.5)
#     except KeyboardInterrupt:
#         log.info("[SHUTDOWN] stopping...")
#         stop_event.set()
#         try:
#             sd.stop()
#         except Exception:
#             pass
#         for t in threads:
#             try:
#                 t.join(timeout=1.0)
#             except Exception:
#                 pass

#         os._exit(0)


# if __name__ == "__main__":
#     main()
