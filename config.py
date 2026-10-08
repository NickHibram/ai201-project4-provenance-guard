"""Fixed scoring and API policy for the local prototype."""

from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = os.getenv('DATABASE_PATH', str(BASE_DIR / 'data' / 'provenance_guard.sqlite3'))
GROQ_MODEL = os.getenv('GROQ_MODEL')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
PROMPT_VERSION = 'v1'
SCORE_CONFIG_VERSION = 'v1'
LLM_TEMPERATURE = 0.1
LLM_TIMEOUT_SECONDS = 20.0
MAX_TEXT_CHARS = 20_000
MAX_CREATOR_ID_CHARS = 100
MAX_REASONING_CHARS = 2_000
MAX_REQUEST_BYTES = 64 * 1024
MIN_WORDS = 50
MIN_SENTENCES = 3
LLM_WEIGHT = 0.60
STYLOMETRIC_WEIGHT = 0.40
LIKELY_HUMAN_MAX = 0.30
LIKELY_AI_MIN = 0.80
SUBMISSION_RATE_LIMIT = '3 per 5 minutes;100 per day'
RATE_LIMIT_STORAGE_URI = 'memory://'
LOCAL_PORT = 7860
CONFIDENCE_SEMANTICS = 'ai_evidence_index_not_probability'
LABELS = {
    'likely_ai': 'Likely AI-generated. This text shows strong indicators of AI generation, but automated detection can be wrong. The creator may appeal this result.',
    'likely_human': 'Likely human-written. This text shows strong indicators of human authorship, though automated analysis cannot verify authorship with certainty.',
    'uncertain': 'Uncertain. Our analysis does not provide enough evidence to confidently determine whether this text is human-written or AI-generated.',
}
APPEAL_NOTICE = 'Under review. The creator has appealed this assessment; it has not been resolved.'
