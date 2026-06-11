#!/usr/bin/env python3

import os
# Suppress warnings and logger outputs for clean console output
os.environ["NEMO_LOG_LEVEL"] = "ERROR"
os.environ["HYDRA_FULL_ERROR"] = "0"

import sys
import time
import json
import logging
import threading
import queue
import collections
import wave
import tempfile
import warnings
import requests
from pathlib import Path
from dotenv import load_dotenv

# Set logging format like local_test.py
logging.basicConfig(level=logging.INFO, format="%(message)s")
warnings.filterwarnings("ignore")
log = logging.getLogger("VOICE")

import numpy as np
import sounddevice as sd
import webrtcvad
import torch
from nemo.collections.asr.models import EncDecRNNTBPEModel
from piper.voice import PiperVoice

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
load_dotenv(PROJECT_ROOT / ".env")

SAMPLE_RATE = 16000
CHANNELS = 1

VAD_MODE = 2
CHUNK_DURATION = 0.03
MIN_SPEECH_SEC = 0.5
SILENCE_SEC = 0.8
MAX_AUDIO_SEC = 20

FORCE_LANG = "en"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a real-time voice assistant speaking over a phone call.
Keep replies natural, human-like and brief and keep it small but full informative.
Give answer in maximum 2 lines, if user ask for more details, then only you can give more information.
Keep your response short and to the point, like a helpful friend on a call but should include all user answer.
Always reply in same language as user.
Use casual spoken tone.
If unclear, ask short clarification.
"""

audio_queue = queue.Queue()
text_queue = queue.Queue()
reply_queue = queue.Queue()

conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
conversation_lock = threading.Lock()

session = requests.Session()

model = None
piper_voices = {}

is_speaking = False

def load_models():
    global model, piper_voices
    
    log.info("[INIT] Loading Parakeet model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"[INIT] Using device: {device}")
    model = EncDecRNNTBPEModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v3",
        map_location=device
    )
    model.eval()
    log.info("[INIT] Parakeet model loaded successfully!")

    log.info("[INIT] Loading Piper TTS voice...")
    voice_file = "en_US-lessac-medium.onnx"
    path = PROJECT_ROOT / "piper" / voice_file
    if path.exists():
        piper_voices["en"] = PiperVoice.load(str(path))
        log.info("[TTS] Loaded English voice (en_US-lessac-medium.onnx)")
    else:
        log.error(f"[TTS] Voice file not found: {path}")

def warmup():
    log.info("[WARMUP] starting")
    
    # Warm up Piper TTS
    voice = piper_voices.get("en")
    if voice:
        try:
            for _ in voice.synthesize("hello"):
                break
        except Exception as e:
            log.warning(f"[WARMUP] TTS warmup failed: {e}")
            
    # Warm up Parakeet STT
    if model:
        try:
            # Create a tiny dummy wav file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_filename = temp_wav.name
            try:
                # 1 second of silence
                dummy_audio = np.zeros(SAMPLE_RATE, dtype=np.int16)
                with wave.open(temp_filename, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(dummy_audio.tobytes())
                model.transcribe([temp_filename], batch_size=1)
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
        except Exception as e:
            log.warning(f"[WARMUP] STT warmup failed: {e}")

    log.info("[WARMUP] done")

vad = webrtcvad.Vad(VAD_MODE)

def is_speech(frame):
    try:
        return vad.is_speech(frame, SAMPLE_RATE)
    except:
        return False

def mic_worker():
    global is_speaking

    while True:
        # Block microphone input if TTS is currently speaking
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
        pre_roll = collections.deque(maxlen=10)  # ~300ms pre-roll
        recording_started = False
        speech_frames = 0
        silence_frames_count = 0

        min_frames = int(MIN_SPEECH_SEC / CHUNK_DURATION)
        silence_timeout_frames = int(SILENCE_SEC / CHUNK_DURATION)

        log.info("🎤 Listening...")

        while True:
            # Stop capturing immediately if TTS starts speaking mid-recording
            if is_speaking:
                buffer = []
                break

            try:
                chunk, _ = stream.read(stream.blocksize)
            except Exception as e:
                log.error(f"[Mic Read Error] {e}")
                break

            frame = chunk.tobytes()
            speech = is_speech(frame)

            if not recording_started:
                if speech:
                    recording_started = True
                    # Initialize buffer with pre-roll frames
                    for p_chunk in pre_roll:
                        buffer.extend(p_chunk)
                    buffer.extend(chunk.flatten())
                    speech_frames = 1
                    silence_frames_count = 0
                else:
                    pre_roll.append(chunk.flatten())
            else:
                buffer.extend(chunk.flatten())
                if speech:
                    speech_frames += 1
                    silence_frames_count = 0
                else:
                    silence_frames_count += 1
                    if silence_frames_count > silence_timeout_frames:
                        break

        stream.stop()
        stream.close()

        # Discard recordings that are too short (noise/clicks)
        if not buffer or speech_frames < min_frames:
            continue

        audio = np.array(buffer, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio) > SAMPLE_RATE * MAX_AUDIO_SEC:
            audio = audio[:SAMPLE_RATE * MAX_AUDIO_SEC]

        meta = {
            "speech_duration": len(audio) / SAMPLE_RATE,
            "speech_end_time": time.time(),
            "stt_time": 0.0,
            "llm_first_token_time": 0.0,
            "llm_total_time": 0.0,
        }
        audio_queue.put((audio, meta))

def stt_worker():
    while True:
        audio, meta = audio_queue.get()
        start = time.time()

        # Convert back to int16 for WAV file writing
        audio_int16 = (audio * 32768.0).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_filename = temp_wav.name

        try:
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_int16.tobytes())

            # Perform transcription using Parakeet (inherently optimized for English output)
            result = model.transcribe([temp_filename], batch_size=1)

            transcription = ""
            if result:
                if isinstance(result, tuple):
                    texts = result[0]
                else:
                    texts = result

                if isinstance(texts, list) and len(texts) > 0:
                    text_item = texts[0]
                    if hasattr(text_item, 'text'):
                        transcription = text_item.text
                    else:
                        transcription = str(text_item)
                else:
                    transcription = str(texts)

            text = transcription.strip()
            
        except Exception as e:
            log.error(f"[STT Error] {e}")
            text = ""
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            audio_queue.task_done()

        if not text or len(text) < 2:
            continue

        stt_duration = time.time() - start
        meta["stt_time"] = stt_duration
        meta["stt_engine"] = "parakeet-tdt-0.6b-v3"
        meta["stt_lang"] = "en"
        meta["text"] = text

        log.info(f"\n🗣️ [USER] {meta['speech_duration']:.2f}s speech → \"{text}\"")
        text_queue.put((text, "en", meta))

def llm_worker():
    while True:
        text, lang, meta = text_queue.get()
        start = time.time()

        user_input = f"{text}\n\n[Reply in {lang}, short spoken.]"

        with conversation_lock:
            conversation.append({"role": "user", "content": user_input})
            conversation[:] = conversation[-12:]

        res = session.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": conversation,
                "temperature": 0.6,
                "max_tokens": 180,
                "stream": True,
            },
            stream=True,
        )

        buffer = ""
        full = ""
        first_token_time = None
        first_chunk_sent = False

        for line in res.iter_lines():
            if not line:
                continue

            line = line.decode("utf-8")

            if line.startswith("data: "):
                if "[DONE]" in line:
                    break

                data = json.loads(line[6:])
                token = data["choices"][0]["delta"].get("content", "")

                if token:
                    if first_token_time is None:
                        first_token_time = time.time()
                        meta["llm_first_token_time"] = first_token_time - start
                    
                    buffer += token
                    full += token
                    threshold = 30 if not first_chunk_sent else 60
                    if len(buffer) > threshold or buffer.endswith((".", "?", "!")):
                        reply_queue.put((buffer, lang, False, meta))
                        buffer = ""
                        first_chunk_sent = True

        if buffer:
            reply_queue.put((buffer, lang, True, meta))

        llm_end = time.time()
        meta["llm_total_time"] = llm_end - start
        meta["full_response"] = full

        log.info(f"🤖 [ASSISTANT] (LLM total: {meta['llm_total_time']:.2f}s) → {full}")
        with conversation_lock:
            conversation.append({"role": "assistant", "content": full})
        text_queue.task_done()

def tts_worker():
    global is_speaking

    stream = sd.OutputStream(samplerate=22050, channels=1, dtype='int16')
    stream.start()

    buffer = ""
    current_lang = "en"

    while True:
        text, lang, is_final, meta = reply_queue.get()

        current_lang = lang
        buffer += text

        should_flush = (
            len(buffer) > 80 or
            buffer.endswith((".", "?", "!")) or
            is_final
        )

        if not should_flush:
            continue

        voice = piper_voices.get("en")
        if not voice:
            buffer = ""
            continue

        is_speaking = True

        tts_synth_start = time.time()
        chunks = voice.synthesize(buffer)

        try:
            first_chunk = next(chunks)
            tts_first_chunk_time = time.time()
            tts_latency = tts_first_chunk_time - tts_synth_start

            if not meta.get("logged_latency"):
                meta["logged_latency"] = True
                overall_latency = tts_first_chunk_time - meta["speech_end_time"]
                llm_ft = meta.get("llm_first_token_time", 0.0)
                log.info(
                    f"  ├─ STT: {meta['stt_time']:.2f}s | LLM: {llm_ft:.2f}s (TTFT) | TTS: {tts_latency:.2f}s (first chunk)\n"
                    f"  └─ Overall Response Latency: {overall_latency:.2f}s"
                )

            audio = np.frombuffer(first_chunk.audio_int16_bytes, dtype=np.int16)
            stream.write(audio)
        except StopIteration:
            pass

        for chunk in chunks:
            audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            stream.write(audio)

        is_speaking = False
        buffer = ""
        reply_queue.task_done()

def main():
    load_models()
    warmup()

    threading.Thread(target=mic_worker, daemon=True).start()
    threading.Thread(target=stt_worker, daemon=True).start()
    threading.Thread(target=llm_worker, daemon=True).start()
    threading.Thread(target=tts_worker, daemon=True).start()

    log.info("\n🟢 System active! Start speaking now...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("\n👋 Exiting...")

if __name__ == "__main__":
    main()