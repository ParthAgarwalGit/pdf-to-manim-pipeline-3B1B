"""Model ids for every paid V2 stage. One place to change them."""

# Gemini 4 Argon: Google's documented latest and most capable developer model (Nov 2026).
TEXT_MODEL = "gemini-4-argon"      # lesson, narration, narration review, brief interpretation
OCR_MODEL = "gemini-4-argon"       # low-confidence page transcription (vision)
CODE_MODEL = "gemini-4-argon"      # Manim generation, revision and screenshot-guided repair
TTS_MODEL = "gemini-3.8-flash-tts"
PRESENTER_MODEL = "gemini-omni-1.2-flash"
# Seconds of audio per output token unit for Gemini TTS pricing: 25 audio tokens per second.
GEMINI_TTS_TOKENS_PER_SECOND = 25
