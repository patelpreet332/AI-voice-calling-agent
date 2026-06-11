# AI Voice Calling Agent

A real-time, multilingual AI voice agent that handles phone calls and local microphone conversations. The system performs speech-to-text transcription, generates intelligent responses via LLM, and synthesizes natural speech — all with sub-second latency and support for English, Hindi, Telugu, Tamil, and Gujarati.

## Project Overview

This project solves the problem of building a low-latency, locally-hosted voice AI pipeline that can:

- **Handle live phone calls** via Twilio Media Streams with real-time bidirectional audio
- **Run entirely on local hardware** without cloud STT/TTS dependencies (Piper TTS + Faster-Whisper)
- **Support Indian languages** using a hybrid STT pipeline (Whisper + IndicConformer-600M)
- **Detect and respond in the caller's language** automatically

Two modes of operation:

1. 📞 **Phone Call Flow** — Twilio → Node.js WebSocket → Python STT/TTS → Twilio
2. 🎙️ **Local Mic Flow** — Microphone → STT → LLM → TTS → Speaker (no Twilio needed)

## Features

- Real-time voice activity detection (VAD) with configurable thresholds
- Hybrid speech-to-text: Faster-Whisper for English, IndicConformer-600M for Indian languages
- Automatic language detection with Hindi/Gujarati and Telugu/English disambiguation
- Streaming LLM responses (Groq + LLaMA 3.1) with chunked TTS for low latency
- Multi-language Piper TTS synthesis (en, hi, te, gu)
- Configurable greeting message on call connect
- Echo suppression via post-speech cooldown

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Node.js + Express + TypeScript |
| **Telephony** | Twilio Voice + Media Streams (WebSocket) |
| **STT** | Faster-Whisper (CPU) + IndicConformer-600M |
| **LLM** | Groq API (LLaMA 3.1 8B Instant) |
| **TTS** | Piper TTS (ONNX, local CPU inference) |
| **API Server** | Python FastAPI + Uvicorn |
| **Audio** | μ-law ↔ PCM conversion, linear resampling |

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     PHONE CALL FLOW                              │
│                                                                  │
│  Twilio ←──WebSocket──→ server.ts (Node.js)                     │
│                              │                                   │
│                    ┌─────────┴─────────┐                        │
│                    │   CallSession      │                        │
│                    │                    │                        │
│              ┌─────┴─────┐   ┌────────┴────────┐               │
│              │    VAD     │   │  Audio Utils     │               │
│              │ (Energy)   │   │ (μ-law ↔ PCM)   │               │
│              └─────┬─────┘   └─────────────────┘               │
│                    │                                             │
│              ┌─────▼─────┐                                      │
│              │ STT Client │──HTTP──→ Python FastAPI (:8000)     │
│              └─────┬─────┘          ├─ /transcribe              │
│                    │                └─ /tts                      │
│              ┌─────▼─────┐                                      │
│              │  LLM       │──HTTPS──→ Groq API                  │
│              │ (Streaming)│                                      │
│              └─────┬─────┘                                      │
│                    │                                             │
│              ┌─────▼─────┐                                      │
│              │ TTS Client │──HTTP──→ Python FastAPI (:8000)     │
│              └─────┬─────┘                                      │
│                    │                                             │
│              PCM → μ-law → Twilio                               │
└──────────────────────────────────────────────────────────────────┘
```

## Project Workflow

### Phone Call Pipeline

1. Twilio places an outbound call and connects Media Streams via WebSocket
2. Incoming μ-law 8kHz audio is decoded to PCM 16kHz
3. VAD detects speech boundaries (start/silence/end)
4. Complete utterance is sent to Python STT service
5. STT auto-detects language, routes to Whisper or IndicConformer
6. Transcribed text is streamed through Groq LLM
7. LLM response chunks are synthesized via Piper TTS
8. TTS audio is resampled to 8kHz μ-law and streamed back to Twilio
9. If the user interrupts, audio playback is immediately stopped

### Local Mic Pipeline

1. Microphone captures audio via `sounddevice`
2. WebRTC VAD detects speech boundaries
3. Audio is transcribed locally using Faster-Whisper / IndicConformer
4. Groq LLM generates a streaming response
5. Response chunks are synthesized via Piper TTS
6. Audio is played through the speaker in real-time

## Project Structure

```
.
├── server.ts                 # Express + Twilio webhook + WebSocket handler
├── lib/
│   ├── call-session.ts       # Per-call pipeline orchestration
│   ├── vad.ts                # Energy-based voice activity detection
│   ├── stt.ts                # STT client (calls Python service)
│   ├── llm.ts                # Groq LLM streaming session
│   ├── tts.ts                # TTS client (calls Python service)
│   ├── audio-utils.ts        # μ-law/PCM conversion + resampling
│   ├── elevenlabs.ts         # ElevenLabs connector (alternative TTS)
│   └── groq.ts               # Groq axios instance
├── python-stt/
│   ├── main.py               # FastAPI: /transcribe + /tts endpoints
│   ├── local_test.py         # Standalone mic → STT → LLM → TTS loop
│   └── requirements.txt      # Python dependencies
├── piper/                    # Piper ONNX voice models (not committed)
├── package.json
├── tsconfig.json
└── .env.example
```

## Installation Guide

### Prerequisites

- Node.js 18+
- Python 3.10+ (3.11 recommended)
- Groq API key ([console.groq.com](https://console.groq.com))
- Twilio Voice-capable account (phone call flow only)
- Microphone + speakers (local mic flow only)

#### System Packages (Local Mic Test)

`sounddevice` requires PortAudio:

- **Ubuntu/Debian:** `sudo apt-get install portaudio19-dev`
- **macOS:** `brew install portaudio`

### Quick Start: Phone Call Flow (Twilio)

#### 1) Install Node dependencies

From the **project root**:

```bash
npm install
```

#### 2) Create Python venv + install deps

```bash
cd python-stt
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

#### 3) Configure environment

From the **project root**, copy and fill `.env`:

```bash
cp .env.example .env
```

Required variables:

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key |
| `TWILIO_ACCOUNT_SID` | Twilio Account SID |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token |
| `TWILIO_PHONE_NUMBER` | Your Twilio phone number |
| `USER_PHONE_NUMBER` | Target phone number for outbound calls |
| `NGROK_URL` | Public HTTPS URL from ngrok |

#### 4) Download Piper voice models

From the **project root**, create the `piper/` directory and download voice models:

```bash
mkdir -p piper
```

**English:**
```bash
wget -O piper/en_US-lessac-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget -O piper/en_US-lessac-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

**Hindi:**
```bash
wget -O piper/hi_IN-priyamvada-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium.onnx
wget -O piper/hi_IN-priyamvada-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium.onnx.json
```

**Telugu:**
```bash
wget -O piper/te_IN-padma-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/te/te_IN/padmavathi/medium/te_IN-padmavathi-medium.onnx
wget -O piper/te_IN-padma-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/te/te_IN/padmavathi/medium/te_IN-padmavathi-medium.onnx.json
```

**Gujarati:**
- Request access and download from [huggingface.co/Arjun4707/piper-gujarati-male](https://huggingface.co/Arjun4707/piper-gujarati-male/tree/main)
- Place `gu_epoch229.onnx` and `gu_epoch229.onnx.json` in `piper/`

#### 5) Start Python STT+TTS service

From the `python-stt/` folder (activate the venv first):

```bash
cd python-stt
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

Sanity check (from any terminal):

```bash
curl http://localhost:8000/health
```

#### 6) Start ngrok + Node server

Open a **new terminal** at the **project root**.

Start ngrok:

```bash
ngrok http 3000
```

Update `NGROK_URL` in `.env` with the HTTPS forwarding URL.

Open another terminal at the **project root** and run:

```bash
npm run dev
```

#### 7) Make a call

Open in browser:

```
http://localhost:3000/make-call
```

This triggers an outbound call from `TWILIO_PHONE_NUMBER` → `USER_PHONE_NUMBER`.

> **⚠️ Twilio Free Trial Note:** If using a free trial account, the call may disconnect after pickup. Answer the call, open the phone keypad, and press any key once to keep it active.

### Quick Start: Local Mic Flow

No Twilio or Node server required.

#### Option A: Default Flow (Whisper / IndicConformer)

1) From the **project root**, configure `.env`:

```bash
cp .env.example .env
```

Required: `GROQ_API_KEY`

Also required: Piper voice models in `piper/` (see step 4 above)

2) From the `python-stt/` folder:

```bash
cd python-stt
source .venv/bin/activate
python3 local_test.py
```

Optional — force a specific language:

```bash
python3 local_test.py hi
```

#### Option B: NVIDIA Parakeet Flow (English-only STT)

The Parakeet flow uses `nvidia/parakeet-tdt-0.6b-v3` for high-accuracy, low-latency English speech-to-text. Since NVIDIA NeMo has specific package requirements (such as PyTorch and ASR toolkit dependencies), we run it in a separate virtual environment (`nemo_env`) to avoid package conflicts.

1) Create and activate `nemo_env`:

From the `python-stt/` folder:

```bash
cd python-stt
python3 -m venv nemo_env
source nemo_env/bin/activate
pip install -U pip setuptools wheel
```

2) Install PyTorch, NeMo ASR, and other dependencies:

```bash
pip install torch torchaudio
pip install "nemo_toolkit[asr]"
pip install sounddevice webrtcvad-wheels piper-tts python-dotenv requests numpy
```

*(Note: Cython may be compiled during NeMo installation. Ensure you have compiler tools installed on your OS, e.g., `sudo apt-get install build-essential` on Ubuntu).*

3) Configure Environment & Piper Voices:
- Verify that `.env` in the project root contains your `GROQ_API_KEY`.
- Ensure the English voice `en_US-lessac-medium.onnx` and its config file are downloaded into the `piper/` folder (see step 4 under Phone Call Flow).

4) Run the Parakeet Local Mic Flow:

With `nemo_env` active:

```bash
python3 local_test_parakeet.py
```

## Environment Variables

See [`.env.example`](.env.example) for the complete list with descriptions.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | — | Groq API key for LLM |
| `TWILIO_ACCOUNT_SID` | Phone flow | — | Twilio Account SID |
| `TWILIO_AUTH_TOKEN` | Phone flow | — | Twilio Auth Token |
| `TWILIO_PHONE_NUMBER` | Phone flow | — | Twilio caller number |
| `USER_PHONE_NUMBER` | Phone flow | — | Target phone number |
| `NGROK_URL` | Phone flow | — | Public HTTPS URL |
| `PORT` | No | `3000` | Node server port |


## Third-Party Services

| Service | Purpose | Required? |
|---------|---------|-----------|
| [Groq](https://groq.com) | LLM inference (LLaMA 3.1 8B) | Yes |
| [Twilio](https://twilio.com) | Phone call routing + Media Streams | Phone flow only |
| [ngrok](https://ngrok.com) | Local tunnel for Twilio webhook | Phone flow only |
| [Hugging Face](https://huggingface.co) | IndicConformer model download | First run only |

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start Node.js server (development) |

## Security Notes

- All API keys and credentials are stored in `.env` (never committed)
- `.env.example` contains safe placeholder values only
- Twilio webhook validates via `NGROK_URL` environment variable
- The Python STT/TTS service binds to `0.0.0.0` — restrict access in production

## Hugging Face / IndicConformer

The Python STT service loads `ai4bharat/indic-conformer-600m-multilingual` via `transformers`. On first run, this will download ~2.4GB of model weights.

If prompted for authentication:

```bash
huggingface-cli login
```

Or set the token:

```bash
export HF_TOKEN="your_token_here"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No default input device` | Check OS audio permissions, ensure PortAudio is installed |
| `IndicConformer failed` | App auto-falls back to Whisper. Set up HuggingFace auth if needed |
| Call connects but no audio | Verify `NGROK_URL` is current HTTPS URL, check Twilio credentials |
| `piper/: No such file or directory` | Run `mkdir -p piper` from project root before downloading voices |


## Screenshots

> Screenshots and demo recordings can be added here.
