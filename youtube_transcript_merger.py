"""YouTube transcript merger with mandatory Webshare proxy + I: drive storage.

Fetches transcripts from YouTube videos using embedded captions when available,
falling back to Whisper (base model, CUDA required) for audio transcription.
All outputs are stored under I:\youtube_pipeline.

Prerequisites
-------------
pip install youtube-transcript-api yt-dlp openai-whisper torch

Environment
-----------
WEBSHARE_PROXY_URL  – Required. The HTTP proxy URL for all network requests.
FFmpeg              – Must be available in PATH for audio extraction.
CUDA GPU            – Required for Whisper fallback transcriptions.
Drive I:            – Must be mounted for temp/output storage.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv("C:\\your_project_folder\\.env")

def _prepend_to_path(path: Path) -> None:
    """Prepend a directory to PATH for the current process."""
    path_str = str(path)
    if path_str and path_str not in os.environ.get("PATH", ""):
        os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")


def resolve_ffmpeg_path() -> Optional[str]:
    """Locate ffmpeg by checking PATH, env hints, and common WinGet locations."""

    detected = shutil.which("ffmpeg")
    if detected:
        return detected

    env_hint = os.getenv("FFMPEG_PATH")
    if env_hint:
        candidate = Path(env_hint).expanduser()
        if candidate.exists():
            _prepend_to_path(candidate.parent)
            return str(candidate)

    local_appdata = os.getenv("LOCALAPPDATA")
    candidates = []
    if local_appdata:
        la_path = Path(local_appdata)
        candidates.extend(
            [
                la_path / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
                la_path
                / "Microsoft"
                / "WinGet"
                / "Packages"
                / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
                / "ffmpeg-8.0.1-full_build"
                / "bin"
                / "ffmpeg.exe",
            ]
        )

    for candidate in candidates:
        if candidate and candidate.exists():
            _prepend_to_path(candidate.parent)
            return str(candidate)

    return None


# torch is imported lazily inside load_whisper_model() to avoid startup failure
# when only using caption-based transcription

try:
    from youtube_transcript_api import (  # type: ignore
        NoTranscriptFound,
        TranscriptsDisabled,
        YouTubeTranscriptApi,
    )
    from youtube_transcript_api.proxies import WebshareProxyConfig  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "youtube-transcript-api is required. Install with 'pip install youtube-transcript-api'."
    ) from exc

try:
    from yt_dlp import YoutubeDL  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("yt-dlp is required. Install with 'pip install yt-dlp'.") from exc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

WEBSHARE_PROXY_URL = os.getenv("WEBSHARE_PROXY_URL")
if not WEBSHARE_PROXY_URL:
    raise SystemExit("Set WEBSHARE_PROXY_URL in the environment before running this script.")

# Build US-specific proxy URL by inserting -US before -rotate in username
# Format: username-rotate:password@host:port -> username-US-rotate:password@host:port
def _build_us_proxy_url(proxy_url: str) -> str:
    """Convert proxy URL to use US country code."""
    parsed = urlparse(proxy_url)
    if parsed.username and "-rotate" in parsed.username:
        us_username = parsed.username.replace("-rotate", "-US-rotate")
        # Rebuild URL with new username
        netloc = f"{us_username}:{parsed.password}@{parsed.hostname}:{parsed.port}"
        return f"{parsed.scheme}://{netloc}"
    return proxy_url

WEBSHARE_PROXY_URL_US = _build_us_proxy_url(WEBSHARE_PROXY_URL)

I_DRIVE_ROOT = Path("I:/")
if not I_DRIVE_ROOT.exists():
    raise SystemExit("Drive I: is not available. Mount it before running the pipeline.")

STORAGE_ROOT = I_DRIVE_ROOT / "youtube_pipeline"
TEMP_DIR = STORAGE_ROOT / "temp"
WHISPER_TMP_DIR = TEMP_DIR / "whisper"
OUTPUT_DIR = STORAGE_ROOT / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "merged_youtube_transcripts.txt"

FFMPEG_PATH = resolve_ffmpeg_path()

RESEARCH_ROOT = STORAGE_ROOT / "research"
TAVILY_OUTPUT_DIR = RESEARCH_ROOT / "tavily"
GEMINI_OUTPUT_DIR = RESEARCH_ROOT / "gemini"

for path in (STORAGE_ROOT, TEMP_DIR, WHISPER_TMP_DIR, OUTPUT_DIR, RESEARCH_ROOT, TAVILY_OUTPUT_DIR, GEMINI_OUTPUT_DIR):
    path.mkdir(parents=True, exist_ok=True)

# Configure persistent logging to disk now that storage root exists
LOGS_DIR = STORAGE_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOGS_DIR / "youtube_transcript_merger.log"
from logging.handlers import RotatingFileHandler
handler = RotatingFileHandler(str(log_file), maxBytes=2_000_000, backupCount=3)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(handler)

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

# Language priority when choosing embedded captions.
PREFERRED_LANGUAGES = ["en", "en-US", "en-GB", "es", "es-419"]

YOUTUBE_RESULT_LIMIT_DEFAULT = 5

# Retry configuration for transient network errors
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 2.0

# Gemini 2.5 Flash rate limit: 5 RPM = 1 request per 12 seconds
# Use 15 seconds between requests for safety buffer
GEMINI_RATE_LIMIT_DELAY = 15

# Cleanup: remove temp audio files older than this many hours
TEMP_CLEANUP_AGE_HOURS = 24

# Lazy-loaded Whisper model reference
_WHISPER_MODEL = None

# External API configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # Set this in your environment or secret store
TAVILY_ENDPOINT = os.getenv("TAVILY_ENDPOINT", "https://api.tavily.com/search")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Set this in your environment or secret store
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # Fast model for writing
GEMINI_RESEARCH_MODEL = os.getenv("GEMINI_RESEARCH_MODEL", "gemini-2.5-flash")  # Fast model for research
GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Kimi API configuration (Moonshot AI - OpenAI compatible) - DISABLED
# Kimi API was unreliable, keeping Gemini-only for now
# KIMI_API_KEY = os.getenv("KIMI_API_KEY")  # Get from https://platform.moonshot.cn/console/api-keys
# KIMI_API_BASE = "https://api.moonshot.cn/v1"
# KIMI_MODEL = os.getenv("KIMI_MODEL", "moonshot-v1-128k")  # 128k context for long research
# Rate limits for Kimi free tier: ~3 requests/min, 1 concurrent request
# KIMI_RATE_LIMIT_DELAY = 25  # seconds between requests to stay under 3/min

REQUESTS_PROXY = {"http": WEBSHARE_PROXY_URL, "https": WEBSHARE_PROXY_URL}
REQUESTS_PROXY_US = {"http": WEBSHARE_PROXY_URL_US, "https": WEBSHARE_PROXY_URL_US}
REQUEST_TIMEOUT_SECONDS = 300  # 5 minutes for large Gemini prompts (270k+ chars)

# Legacy Tavily prompt (kept for reference)
# RESEARCH_PROMPT_TEMPLATE = (
#     "As an expert in {topic}, research and analyze {topic}. Identify, categorize, and analyze the subject. "
#     "The final report should present an overall framework, detailed category discussions, practical implementation methods, "
#     "future trends, and recommendations. Conduct the study and prepare a report. Use the provided notes verbatim when helpful "
#     "and fact-check all findings.\n\nProvided notes:\n{notes}\n"
# )

# Gemini Deep Research prompt - comprehensive analysis
GEMINI_RESEARCH_PROMPT = """You are an expert researcher and analyst. Your task is to produce a COMPREHENSIVE, DETAILED research report of AT LEAST 3,000 words.

## TOPIC TO RESEARCH:
{topic}

## YOUR MISSION:
Analyze ALL the provided source materials thoroughly. Extract every relevant fact, statistic, quote, insight, and perspective. Your research report will be used to write a documentary script, so completeness is critical.

## REQUIRED OUTPUT STRUCTURE (follow this exactly):

### 1. EXECUTIVE SUMMARY (300+ words)
- Main thesis and key findings
- Most important statistics and data points
- Critical insights that emerged from the sources

### 2. CURRENT LANDSCAPE (500+ words)
- What is happening right now in detail
- Key players, companies, and their market positions
- Recent news, developments, and announcements
- Market size, growth rates, and key metrics

### 3. HISTORICAL CONTEXT & EVOLUTION (400+ words)
- How this space/topic developed over time
- Key milestones and turning points
- Lessons from past cycles or events
- What patterns repeat

### 4. DETAILED ANALYSIS BY CATEGORY (600+ words)
- Break down the topic into logical subcategories
- Analyze each subcategory with specific data
- Compare and contrast different options/approaches
- Include specific names, numbers, and facts

### 5. RISKS, CONCERNS & BEAR CASE (400+ words)
- What could go wrong - be specific
- Warning signs and red flags
- Regulatory concerns
- Competitive threats
- What skeptics and critics say

### 6. OPPORTUNITIES & BULL CASE (400+ words)
- Growth catalysts and tailwinds
- What optimists see
- Underappreciated factors
- Potential upside scenarios

### 7. PRACTICAL IMPLEMENTATION (300+ words)
- How someone would actually approach this
- Step-by-step considerations
- Best practices from the sources
- Common mistakes to avoid

### 8. EXPERT OPINIONS & QUOTES (200+ words)
- Direct quotes from sources (attribute them)
- Points where experts agree
- Points where experts disagree
- Credibility notes

### 9. FUTURE OUTLOOK & TRENDS (300+ words)
- Where this is heading
- Emerging trends
- Predictions from sources
- What to watch for

### 10. KEY FACTS & STATISTICS (bullet list)
- Every important number from the sources
- Dates and timelines
- Names and companies mentioned
- Specific data points for the script writer

## CRITICAL REQUIREMENTS:
- Use SPECIFIC numbers, percentages, dates, and names from the sources
- Quote sources directly when they say something insightful
- Note when sources contradict each other
- Include ALL relevant information - do not summarize or shorten
- This report must be AT LEAST 3,000 words
- The script writer depends on this research being thorough

INFLATION REALITY CHECK:
- Flag whether numbers are nominal or inflation-adjusted. Prefer real terms when projecting 10+ years.
- For US figures, assume ~3% long-run inflation unless specified; for LATAM, note higher/volatile inflation when relevant.
- When sources project future dollar amounts, add a quick note like "~55% of that in today's dollars after 20 years at ~3% inflation" when helpful.

## GLOBAL AUDIENCE:
The audience is GLOBAL, not US-only. When researching:
- Flag US-specific content (401k, Roth IRA, US tax rules) so the script writer knows to generalize it
- Focus on universally applicable investment concepts
- US markets/stocks/ETFs are fine (globally accessible), but US-only life situations should be noted

## SOURCE MATERIALS TO ANALYZE:
{notes}
"""

# Spanish translation prompt - neutral Latin American Spanish for YouTube
GEMINI_TRANSLATION_PROMPT = """You are a professional translator specializing in YouTube video scripts.

TRANSLATE the following English script to NEUTRAL LATIN AMERICAN SPANISH.

CRITICAL RULES:
1. Use "tú" (informal you), NOT "usted" or "vos"
2. Use neutral Latin American vocabulary - avoid Spain-specific terms (e.g., use "computadora" not "ordenador", "celular" not "móvil")
3. Keep the same conversational, spoken tone - this is for a teleprompter
4. Preserve all numbers, percentages, and company names exactly
5. Keep English terms that are commonly used in finance/tech (ETF, S&P 500, etc.)
6. Maintain sentence rhythm - short punchy sentences stay short
7. Do NOT add content - translate what's there
8. No headers, no formatting, no notes - just the translated script

REMOVE OR ADAPT US-ONLY CONTENT:
This version is for LATIN AMERICAN audience. Remove or replace:

- 401(k), Roth IRA, Traditional IRA, HSA → "cuentas de jubilación con ventajas fiscales"
- "IRS", "tax-deferred", "tax-free growth" → "ventajas fiscales según tu país"
- Specific US tax law references → DELETE entirely
- "Schwab", "Fidelity", "Vanguard" as brokers → "tu bróker" (ETF names are OK to keep)
- "Social Security" → DELETE or "pensión gubernamental"
- US minimum wage context → Use percentages or neutral context

PALABRAS PROHIBIDAS (busca y elimina/reemplaza):
- "Roth" → ELIMINAR o reemplazar por "con ventajas fiscales"
- "HSA" / "Health Savings Account" → ELIMINAR párrafo completo
- "59 años y medio" / "59.5" → ELIMINAR (edad de retiro de EE.UU.)
- "IRS" → ELIMINAR
- "backdoor" / "mega backdoor" → ELIMINAR
- "Required Minimum Distribution" / "RMD" → ELIMINAR

ADAPTACIÓN DE CIFRAS PARA LATAM:
- Si el guion menciona "$1M para retirarse" → USA "$300,000-450,000 USD"
- Si dice "$4,000/mes para vivir" → USA "$1,000-1,500 USD/mes"
- Contexto: Salario mínimo México ~$473 USD/mes, Colombia ~$431 USD/mes
- REGLA: Cifras de retiro basadas en costos de EE.UU. → DIVIDE entre 2-3

EJEMPLOS MAL vs BIEN:
MAL: "El ejemplo de Peter Thiel con su Roth IRA de $5 mil millones..."
BIEN: "El ejemplo de Peter Thiel ilustra cómo las cuentas con ventajas fiscales pueden crecer agresivamente si se aprovechan desde joven."

MAL: "Las cuentas de ahorro para la salud con ventajas fiscales (HSA)..."
BIEN: ELIMINAR párrafos sobre HSA (no existen en LATAM)

MAL: "accesibles a los 59 años y medio"
BIEN: ELIMINAR esta frase (es edad específica de EE.UU. para 401k)

KEEP:
- ETF names (SCHD, VYM, DGRO) - globally accessible
- Universal concepts (dividends, compound interest, diversification)
- Company examples (Apple, Coca-Cola, etc.)

ENGLISH SCRIPT TO TRANSLATE:
{script}
"""

# Update this prompt to adjust the Gemini writing style.
GEMINI_MASTER_PROMPT = """YOUTUBE FINANCE SCRIPT PROMPT

ROLE: You are a retail investor AND objective analyst. You share your personal journey while letting data speak for itself. You are NOT a financial advisor, guru, or teacher.

MISSION: Write a script based strictly on the research provided. Pure substance. No filler.

TARGET LENGTH: 1,700-2,500 words. This is a HARD limit. If over 2,500, cut ruthlessly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GLOBAL AUDIENCE – NO US-ONLY CONTENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The audience is GLOBAL. EXCLUDE US-only life situations:
- 401(k), Roth IRA, HSA → "tax-advantaged retirement accounts"
- US tax rules → "tax implications vary by country"
- Social Security, Medicare, FICA → REMOVE
- US-specific brokers → "your broker"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEGAL SAFETY: THE JOURNAL + ANALYST BLEND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is the most important rule. NEVER give financial advice or instructions.

THE BLEND:
- Use ATTRIBUTION MODE for facts: "The data shows," "Historically," "The math suggests"
- Use JOURNAL MODE for decisions: "I decided," "For my portfolio," "What worried me was"

ABSOLUTE BAN ON IMPERATIVES:
- NEVER use: "You should," "You need to," "Buy this," "Sell that," "Allocate X%"
- NEVER use: "Make sure to," "Don't forget to," "Start by..."

ATTRIBUTION MODE (for facts and data):
Let the data speak. Don't claim opinions as facts.

BAD (Opinion as fact): "This is a great company."
GOOD (Attribution): "The company increased free cash flow 20% year-over-year."

BAD (Advice): "This ETF is perfect for growth."
GOOD (Attribution): "Historically, investors have used this ETF for growth exposure."

JOURNAL MODE (for decisions and reactions):
Frame YOUR choices as personal, not universal advice.

BAD (Advice): "Buy SCHD for stability. Put 15% of your portfolio here."
GOOD (Journal): "I bought SCHD because I wanted stability. I put about 15% of my portfolio here."

BAD (Instruction): "Avoid high fees by using Limit Orders."
GOOD (Journal): "I noticed high fees were eating my profits, so I switched to Limit Orders."

THE BLEND IN ACTION:
"The data shows Tesla's P/E is 326." (Attribution - fact)
"That makes me uncomfortable at this valuation." (Journal - personal reaction)
"Investors seeking aggressive growth might see it differently." (Attribution - let viewer decide)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VOICE & TONE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PERSPECTIVE: First-person for decisions ("I," "My"). Third-party for facts ("The data shows").

STYLE: Evidence-based but personal. Think "Bloomberg Documentary" meets "talking to a friend."

KEYWORDS: "The data suggests," "Historically," "The tradeoff is," "For my situation," "What concerned me was"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INFLATION REALITY CHECK (MANDATORY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALWAYS clarify when numbers are nominal vs inflation-adjusted.
- When projecting 10+ years, add "in today's purchasing power" and show a rough real equivalent.
- US inflation baseline: ~3% long run (recent spikes higher). Quick mental decay: ~55% of today's value after 20 years, ~41% after 30.
- If citing $X in 30 years, add "about $Y in today's dollars" (use ~3% unless specified).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GLOBAL AUDIENCE – NO US-ONLY CONTENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ATTITUDE: Skeptical until proven otherwise. "Looking under the hood." Boring is beautiful.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE GOLDEN RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"If it sounds like writing, rewrite it."

Read every sentence out loud. If you run out of breath, split it. If you wouldn't say it to a friend, cut it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SENTENCE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SENTENCE LENGTH:
- Target: 8-12 words (the sweet spot for spoken clarity)
- Most sentences: under 20 words
- Hard limit: 30 words (only for complex explanations that can't be split)
- Mix short punches (3-6 words) with medium explanations (10-15 words)

THE BREATH TEST:
Read it aloud. If you inhale mid-sentence, it's too long.

RHYTHM PATTERN (THE SWEET SPOT):
Keep sentences SHORT and EASY TO READ, but vary the length to sound natural.

TARGET: Most sentences 6-12 words. Occasional punch (3-5 words). Occasional flow (13-16 words max).

GOOD RHYTHM EXAMPLE:
"Here's the problem." (3 words - punch)
"Offshore exchanges charge lower fees but carry regulatory risk." (9 words - explain)
"That's the trade-off." (3 words - land it)
"So you need to decide what matters more to you." (10 words - flow)

BAD (staccato - too choppy):
"Tesla has 12 goals. Each one is tied to a target. He needs both keys. The first key is market cap. The second key is operations."

BAD (too long):
"It fundamentally changed the question from 'Which exchange has the lowest fees?' to 'Where is our money actually safe?' because we aren't just looking for trades anymore." (28 words)

GOOD (balanced):
"Tesla set 12 goals. Each one has two requirements: a market cap target and an operational target. Hit both, and Musk earns that batch of shares."

THE RULE: Alternate between short (3-6 words) and medium (7-12 words). Use longer sentences (13-16 words) sparingly—only when combining two related ideas.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOOK RULES (FIRST 10 SECONDS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Under 10 words. Concrete anchor. No preamble.

PRIORITY ORDER (use the first type that fits your topic):

1. SHOCKING NUMBER: "FTX lost $8 billion in customer funds."
2. DIRECT QUESTION: "Which crypto exchange won't lose your money?"
3. SURPRISING FACT: "90% of trading volume happens on unregulated exchanges."

MORE HOOK STYLES TO CHOOSE FROM:

CONTRARIAN: "Everyone says buy the dip. Here's why that's wrong."
RESULTS-BASED: "These 3 ETFs grew my net worth 15% this year."
WISDOM/REGRET: "What I wish someone told me about money 10 years ago."
MARKET MOVEMENT: "Bitcoin just hit $100K. What happens next?"
RELATABLE: "Everyone's panicking about stocks dropping."

HOOK STRUCTURE:
1. Hook (under 10 words)
2. IMMEDIATELY follow with key data that validates the hook
3. Then state your answer

NEVER START WITH:
- "I've been thinking about..."
- "It's interesting to note that..."
- "Today we're going to talk about..."
- "Let me take you back to..."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRUCTURE: ANSWER FIRST, EXPLAIN LATER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is NOT an essay. It's a tutorial.

WRONG (essay logic):
Problem -> History -> Context -> Philosophy -> Solution (buried at 75%)

RIGHT (tutorial logic):
Hook -> Solution -> Why it works -> How to do it -> Edge cases

THE VIEWER MUST KNOW YOUR MAIN THESIS BY THE 60-SECOND MARK.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCRIPT STRUCTURE (1,800-2,200 WORDS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 1: THE HOOK (~150 words / ~1 minute)
- Hook: Under 10 words, concrete fact or shocking number
- The Pivot: Connect it to your personal situation
- Your Thesis: "Here is exactly what I am doing about it."

CRITICAL: By the 60-second mark, viewers know your thesis and why it matters to you personally.

SECTION 2: THE BULL CASE - "Why This Could Work" (500-650 words / ~4 minutes)
Present the strongest arguments FOR the investment. Use Attribution Mode.

- "The main argument for this investment is..."
- "The data shows..."
- "Historically, this has performed..."
- "What caught my attention was..."

Be fair to the bull case even if you're skeptical. Present the best evidence.

SECTION 3: THE BEAR CASE - "What Could Go Wrong" (400-550 words / ~3 minutes)
Present the risks and counterarguments. This builds credibility.

- "But here's what the brochure doesn't tell you."
- "The risk that concerns me most is..."
- "Critics point out that..."
- "The math breaks down if..."

Be honest about real downsides. Don't oversell.

SECTION 4: MY TAKE - "What I'm Actually Doing" (400-500 words / ~3 minutes)
Now switch to Journal Mode. Share YOUR personal decision.

- "Given all this, here's what I decided..."
- "For my portfolio, I allocated about X% because..."
- "For my risk tolerance, I preferred X over Y."
- "I could be wrong if..."

SECTION 5: CLOSE (~100 words / ~1 minute)
- Brief summary of your main takeaway
- "For my portfolio, I'm doing X."
- No calls-to-action. Just end on a strong thought.

TRANSITIONS BETWEEN SECTIONS:
One sentence maximum. Transitions must be SUBSTANTIVE, not structural.

BAD (just announcing the next section):
- "Now let's talk about fees."
- "Now that we've covered X, let's look at Y."
- "Moving on to the risks..."

GOOD (bridge the concepts with substance):
- "But low fees don't matter if you can't withdraw your money."
- "The numbers look good on paper. Here's where they break down."
- "That's the upside. Here's what could go wrong."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANGUAGE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ELIMINATE:
- Metaphors: "navigating a minefield," "fortress," "bazaar"
- Capitalized concepts: "Onshore Fortress," "Trading Floor"
- Academic words: "fundamentally," "paradigm," "catastrophic"
- Hedging: "might," "perhaps," "around," "approximately"
- Excessive "we/us": Don't manufacture tribal identity
- Vague "this" statements: "This indicates," "This shows," "This means," "This will lead to"

VAGUE PRONOUN FIX:
Never start a sentence with "This" followed by a generic verb. Name the subject.

BAD: "This shows the market is overvalued."
GOOD: "The 35x P/E ratio shows the market is overvalued."

BAD: "This indicates strong growth."
GOOD: "Revenue jumping 40% indicates strong growth."

BAD: "This will lead to higher prices."
GOOD: "The supply shortage will push prices higher."

Ask yourself: "What is 'this'?" If you can't point to a specific noun in the previous sentence, rewrite it.

ACTIVE VOICE RULE:
Use active voice. The subject does the action. Passive voice buries the actor and sounds weak.

BAD: "The stock was bought by Buffett."
GOOD: "Buffett bought the stock."

BAD: "The dividend was increased by 10%."
GOOD: "The company raised the dividend 10%."

BAD: "A decision was made to cut costs."
GOOD: "Management decided to cut costs."

If you see "was [verb]ed by" or "has been [verb]ed," rewrite it.

REPLACE WITH:
- Literal terms: "regulated exchange" not "fortress"
- Plain English: "big problem" not "catastrophic failure"
- Exact numbers: "1.49% fee" not "premium pricing"
- Direct address: "you" not "we as investors"
- Confident statements: "This costs $14.90" not "This might cost around $15"

TECHNICAL TERMS:
Define once on first use in simple language.
Example: "Proof-of-reserves—that's when an exchange proves they actually hold your crypto—can be verified at..."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SENSITIVE TOPICS (ONLY WHEN RELEVANT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SKIP THIS SECTION if your topic is purely technical (ETF comparison, stock analysis).
APPLY THIS SECTION if your topic touches: gender/racial wealth gaps, war profiteering, surveillance tech, pharma pricing, labor exploitation, predatory lending, or environmental ethics.

THE CORE RULE:
Lead with the systemic gap. Don't lead with stats that sound like criticism of marginalized groups.

BAD: "Nine out of ten richest women inherited their money."
(Sounds like: "Women didn't earn it")

GOOD: "Only 13% of billionaires are women. Half the population, 13% of the wealth."
(Sounds like: The system has a problem)

HOW TO HANDLE SENSITIVE TOPICS:
- The inequality IS the hook. Lead with the gap.
- Analyze SYSTEMS, not individuals.
- Celebrate those who broke through despite barriers.
- Ask "why does this gap exist?" with curiosity, not preaching.
- Connect to practical investor implications.

DON'T:
- Hedge with "This isn't a criticism..."
- Get preachy or lecture
- Make it partisan political
- Ignore obvious systemic issues

TONE: Pro-equality, pro-opportunity, direct. Finance content, not activism.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR VOICE & ORIGINAL INSIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PERSONALITY: "The Educator, Not the Hype-Man"
Think Patrick Boyle, The Plain Bagel, Joseph Carlson, Matt Derron, Rose Han.

DELIVERY STYLE:
- Calm competence. Measured, articulate, never shouting.
- Structured thinking. Speak in frameworks, steps, and lists.
- Rational optimist. Positive about wealth-building, skeptical of "get rich quick."
- Confidence from data and reading, not market speculation.

EMOTIONAL HOOK: "Ambitious but Efficient"
- Efficiency over effort. Working smart beats grinding.
- Abundance mindset. Focus on multiplying wealth, not hoarding pennies.
- "Boring is beautiful." Reject the casino mentality.

ORIGINAL INSIGHT REQUIREMENT:
Your script MUST contain at least one original synthesis:
- Connect dots others haven't linked
- Challenge conventional wisdom with evidence
- Find second-order effects others miss
- Contrarian views backed by proof

Voice it naturally:
"Here's what I noticed that isn't getting attention..."
"When you connect these pieces, something interesting emerges..."
"Most people focus on X. But the real story is Y."

YOUR PERSPECTIVE:
- Share genuine reactions: "Honestly, this surprised me."
- Be specific about your approach: "For my portfolio, I'm doing X because..."
- Show your reasoning: "Here's why I lean this way..."
- Turn scary concepts into simple systems. Make the complex feel accessible.

WHAT THIS IS NOT:
- Hype ("To the moon!" or panicked urgency)
- Penny-pinching content (extreme frugality, saving coins in jars)
- Gambling mentality (differentiate strategy from speculation)
- Guru positioning ("my secret method")
- False humility ("I'm just a regular guy but...")

The goal: Sound like a calm, knowledgeable educator with a clear point of view.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTENT PRIORITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRIORITIZE IN THIS ORDER:

1. SPECIFIC DATA (majority of script):
- Specific numbers (fees, limits, volumes, percentages)
- Direct comparisons (Platform A vs B on exact metrics)
- Step-by-step instructions (First do X, then Y)
- Concrete examples (real situations, real outcomes)

2. BRIEF CONTEXT (supporting role):
- One mention of relevant history, max 3 sentences
- "Why this matters" tied to specific actions

3. PHILOSOPHY/FEELINGS (minimal):
- Intro and outro only
- Cut entirely if running long

THE "SO WHAT?" TEST:
After every paragraph, ask: "What can the viewer DO with this?"
If the answer is "understand context" or "feel informed," CUT IT.
Only keep content where the answer is "take this action" or "make this decision."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HISTORY & CONTEXT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Historical context is only allowed if it DIRECTLY supports actionable advice.

BAD: "Let me tell you about Mt. Gox..." (story for its own sake)
GOOD: "After Mt. Gox, Kraken added proof-of-reserves. You can verify your funds at [link]. Here's how..."

Every historical reference must end with a specific action or recommendation.
Maximum: 3 sentences per historical reference. If it needs more, cut it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT ANTI-REPETITION POLICY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LINEAR EXPLANATION:
Explain a concept (vesting, dividends, P/E ratio) FULLY only once. Define it, explain how it works, then NEVER re-introduce or re-summarize it. Assume the viewer remembers.

MAXIMUM 2 MENTIONS:
If a concept is complex, you may mention it twice—once to explain, once to apply. That's it. No third time.

NO LOOPING:
If you've stated a mechanism or argument, move forward immediately. Don't circle back "for emphasis."

BAD (Musk script repeated "vesting" explanation 4 times):
- "Vesting means getting paid with shares over time."
- Later: "The vesting period is very important. Vesting means..."
- Later: "Once shares are 'earned,' they don't vest immediately..."
- Later: "They vest on the 7.5-year anniversary..."

GOOD:
- "Vesting means Musk earns shares over 7.5 to 10 years. He can't sell until 2038." (Done. Never explain vesting again.)

NARRATIVE FLOW + FLUFF CUTS (GENERAL):
- ONE AND DONE: Once you explain a concept, do NOT re-explain it later. If mentioned again, keep it to a single callback line without redefining.
- MERGE-THEN-MOVE-ON: If two paragraphs land on the same takeaway, MERGE into one or DELETE the weaker one. No twin endings.
- FORWARD ONLY: Each topic gets ONE home. Later mentions = one-line reference, no rehash. No circling back "for emphasis."
- SUBSTANTIVE TRANSITIONS ONLY: Ban empty pivots like "Now let's talk about..." Every bridge must add a fact, contrast, or consequence.
- DATA DENSITY CAP: Max 3 numbers per paragraph. If more, collapse to a range and keep the most dramatic figure.
- PARAGRAPH PURPOSE: Each paragraph must do exactly one job—claim, evidence, or consequence/action. If it tries to do all three, split or cut.
- FLUFF KILL LIST: Delete sentences that are pure feelings without evidence, meta-commentary about the script, or generic caveats ("everyone is different," "it depends").
- POST-GEN SCAN: Hunt for repeated noun+number patterns; collapse them into one summary sentence and delete the duplicates.

ETF CONSOLIDATION (SPECIFIC):
All similar ETFs in ONE sentence. Not separate paragraphs.
"SCHD, DGRO, and VYM offer diversified dividend exposure with fees under 0.1%." DONE.
Do NOT then list each ETF again with the same stats.

LINEAR STRUCTURE – NO CIRCLING BACK:
The script moves FORWARD only. Once a topic is covered, it is CLOSED.
- If you explained DRIPs in paragraph 5, do NOT mention DRIPs again in paragraph 12.
- If you listed ETF fees once, do NOT revisit fees later.
- If you defined a term, do NOT define it again "for clarity."
- Every paragraph must introduce NEW information. If it doesn't, DELETE it.
Mental test: Draw an arrow from each paragraph to what it references. If any arrow points BACKWARD to earlier content, rewrite or cut.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA PRESENTATION RULES (BROADCAST STYLE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE RULE OF 3:
Never include more than 3 numbers in a single paragraph. If you have more data, use a bulleted list OR select only the most dramatic figure.

LIST EXCEPTION:
If the script's TITLE or TOPIC is a specific list ("10 best dividend stocks," "7 ETFs for retirement"), keep ALL items in that list. The list IS the content. But for general examples proving a point ("dividend stocks work"), limit to 3 varied examples.

ROUND FOR THE EAR, NOT THE EYE:
Format numbers so they're easy to SAY and easy to REMEMBER.

BAD: "$2,665,555" or "9.84%" or "$174,494"
GOOD: "2.6 million" or "nearly 10 percent" or "about 175 thousand"

RULE: If you wouldn't say the exact number to a friend, round it.

SKIP INTERMEDIATE MILESTONES:
Don't list every step (10, 20, 30, 40 years). Pick only the START and the END, or the single most impressive milestone.

BAD (robotic data dump from dividend script):
"After 10 years, your investment value reaches almost $26,000. Your annual income is $1,297. After 20 years, your annual income grows to $3,364. After 30 years, it hits $8,725 annually. Your investment value reaches $174,494. After 40 years, it peaks at $22,630 annually. Your investment value reaches $452,593."

GOOD (broadcast style):
"Start with $10,000 and don't add another penny. After 40 years, your portfolio grows to over $450,000. Now imagine you add $5,000 every year. By year 40, your total investment skyrockets past $2.6 million. That's the power of compounding."

WHY IT WORKS:
- Rounds $452,593 to "over $450,000" (easier to remember)
- Skips years 10, 20, 30 (noise) and goes straight to year 40 (the payoff)
- Only 2-3 key figures instead of 8
- Uses narrative connectors ("Now imagine") instead of robotic lists

SIMPLE SENTENCE STRUCTURE:
Use Subject-Verb-Object order. Helps non-native speakers and sounds more natural when spoken.

BAD: "What you buy with $2 trillion that you can't already buy with $1 trillion is the question Makunda raises."
GOOD: "Makunda asks a fair question. What does $2 trillion buy that $1 trillion doesn't?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RED FLAGS TO CUT IMMEDIATELY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Any sentence starting with:
- "I've been thinking about..."
- "It's interesting to note that..."
- "We need to understand that..."
- "Let me take you back to..."

Any paragraph that:
- Describes feelings instead of facts
- Repeats a point without adding new information
- Contains zero actionable content
- Uses metaphors instead of literal language

Any section that:
- Builds "tension" before revealing information
- Describes the "emotional weight" of a decision
- Treats a simple choice as an existential dilemma

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE PROHIBITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NEVER use:
- Hype: "To the moon," "This is going to explode"
- Guru positioning: "My secret strategy," "What nobody tells you"
- Fear-mongering: "Everything is going to crash"
- False urgency: "You need to act NOW"
- Manufactured drama: "catastrophic," "devastating," "brutal" (unless literally true)
- Collective identity: Excessive "we're all in this together"
- Production directions: (V.O.), [B-roll], (pause)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FACT-CHECKING MANDATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL:
- ONLY use facts, figures, and data from your research sources
- DO NOT invent statistics, percentages, or dollar amounts
- If a specific number isn't in your research, don't guess—use qualitative language
- Be precise: "1.49%" not "around 1.5%"

IF RESEARCH IS INSUFFICIENT:
Say "Based on available information..." and stick to what you know.
Never invent data to fill gaps. It's better to be vague than wrong.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TELEPROMPTER-READY. Pure spoken words. Nothing else.

No scene descriptions. No stage directions. No labels. No visual cues.
Just the words to read on camera.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUALITY CHECKLIST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before submitting, verify:

[ ] Total word count between 1,800-2,200 words? (HARD limit)
[ ] JOURNAL + ANALYST BLEND: Decisions in first-person, facts attributed?
[ ] No imperatives ("You should," "Buy this," "Make sure to")?
[ ] Hook under 10 words with concrete anchor?
[ ] Thesis stated by 60-second mark?
[ ] Most sentences under 20 words? (30-word max for complex explanations only)
[ ] Sentence rhythm varies (mix short + medium, no staccato)?
[ ] All facts from research sources (no invented data)?
[ ] Numbers formatted for the EAR ("2.6 million" not "$2,665,555")?
[ ] Max 3 numbers per paragraph (Rule of 3)?
[ ] Each concept explained ONCE, max twice? No re-definitions?
[ ] Bull Case and Bear Case both fairly presented?
[ ] Transitions are substantive, not structural?
[ ] No metaphors or academic language?
[ ] Passes the "read aloud" test?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Only the spoken script. No headers. No formatting. No section labels.
Just the words to read on camera.
"""

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# Concurrency defaults
DEFAULT_METADATA_CONCURRENCY = int(os.getenv("YT_CONCURRENCY", "4"))
DEFAULT_WHISPER_WORKERS = int(os.getenv("WHISPER_WORKERS", "1"))
ALLOW_WHISPER_CPU = os.getenv("WHISPER_ALLOW_CPU", "false").lower() in ("1", "true", "yes")


def _build_ydl_opts(**overrides):
    """Base yt_dlp options with Webshare proxy enabled."""
    opts = {
        "quiet": True,
        "skip_download": True,
        "noplaylist": True,
        # Suppress yt-dlp warnings about SABR streaming (known YouTube limitation, see yt-dlp #12482)
        "no_warnings": True,
    }
    if WEBSHARE_PROXY_URL:
        opts["proxy"] = WEBSHARE_PROXY_URL

    extractor_overrides = overrides.pop("extractor_args", None)
    opts.update(overrides)
    if extractor_overrides:
        base_args = opts.setdefault("extractor_args", {})
        for key, value in extractor_overrides.items():
            if isinstance(value, dict) and isinstance(base_args.get(key), dict):
                base_args[key].update(value)
            else:
                base_args[key] = value
    return opts


def log_environment_summary(run_research: bool) -> None:
    """Log key environment prerequisites and warn about missing optional secrets."""
    if FFMPEG_PATH:
        logger.info("FFmpeg detected at %s", FFMPEG_PATH)
    else:
        logger.warning("FFmpeg is not in PATH. yt_dlp audio extraction will fail without it.")

    if not WEBSHARE_PROXY_URL:
        logger.error("WEBSHARE_PROXY_URL is unset; networking requirements will fail.")
    else:
        logger.info("Using Webshare proxy: %s", WEBSHARE_PROXY_URL)

    if not I_DRIVE_ROOT.exists():
        logger.error("Drive I: is not available. Mount it before running the pipeline.")
    else:
        logger.info("Detected I: drive for storage (%s)", I_DRIVE_ROOT)

    if run_research:
        if not TAVILY_API_KEY:
            logger.warning("TAVILY_API_KEY is missing; research phase will fail.")
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY is missing; Gemini phase will fail.")
    else:
        if not TAVILY_API_KEY or not GEMINI_API_KEY:
            logger.info("Tavily/Gemini API keys not set. Set them before running with --run-research.")



@dataclass
class TranscriptResult:
    """Container for a single video transcript."""
    title: str
    url: str
    source: str  # "captions" or "whisper"
    text: str
    video_id: str = ""
    duration_seconds: int = 0
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "topic"


def current_timestamp() -> str:
    return datetime.utcnow().strftime(TIMESTAMP_FORMAT)


def format_duration(seconds: int) -> str:
    """Format duration in seconds to HH:MM:SS or MM:SS."""
    if seconds <= 0:
        return "unknown"
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def build_notes_from_transcripts(transcripts: List[TranscriptResult]) -> str:
    sections: List[str] = []
    for result in transcripts:
        sections.append(
            "\n".join(
                [
                    f"### {result.title}",
                    f"URL: {result.url}",
                    f"Source: {result.source}",
                    f"Duration: {format_duration(result.duration_seconds)}",
                    result.text.strip(),
                ]
            )
        )
    return "\n\n".join(sections)


def compose_research_notes(transcripts: List[TranscriptResult], additional_text: str = "") -> str:
    """Compose all research notes with clear section labels.
    
    Returns combined notes with:
    - Search engine scraped content (from additional_text)
    - YouTube transcripts (from transcripts list)
    """
    parts = []
    
    # Add search engine content first (if any)
    if additional_text.strip():
        parts.append(additional_text.strip())
    
    # Add YouTube transcripts with clear header
    youtube_notes = build_notes_from_transcripts(transcripts)
    if youtube_notes.strip():
        youtube_section = "\n".join([
            "=" * 60,
            "=== YOUTUBE VIDEO TRANSCRIPTS ===",
            "=" * 60,
            youtube_notes.strip(),
        ])
        parts.append(youtube_section)
    
    return "\n\n".join(parts)


def read_additional_notes(notes_path: Optional[Path]) -> str:
    if not notes_path:
        return ""
    try:
        text = notes_path.read_text(encoding="utf-8")
        logger.info("Loaded supplemental notes from %s", notes_path)
        return text
    except FileNotFoundError:
        raise SystemExit(f"Additional notes file '{notes_path}' was not found.") from None


def _request_with_retry(url: str, *, method: str = "post", retries: int = MAX_RETRIES, use_proxy: bool = False, **kwargs) -> requests.Response:
    """Make an HTTP request with retry logic.
    
    Handles 429 rate limit errors with longer delays for Gemini API.
    Proxy is only used when use_proxy=True (for web scraping, not for API calls).
    """
    requester = getattr(requests, method.lower())
    kwargs.setdefault("timeout", REQUEST_TIMEOUT_SECONDS)
    if use_proxy:
        kwargs.setdefault("proxies", REQUESTS_PROXY)
    
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = requester(url, **kwargs)
            
            # Special handling for 429 rate limit
            if response.status_code == 429:
                # Exponential backoff: 30s, 60s, 120s, 240s, 480s
                wait_time = 30 * (2 ** (attempt - 1))
                logger.warning("Rate limited (429) on attempt %d. Waiting %d seconds before retry...", attempt, wait_time)
                time.sleep(wait_time)
                if attempt < retries:
                    continue
                    
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < retries:
                # Use exponential backoff for rate limit issues
                delay = RETRY_DELAY_SECONDS * attempt
                if "429" in str(exc) or "Too Many Requests" in str(exc):
                    delay = 30 * (2 ** (attempt - 1))  # 30s, 60s, 120s, 240s
                logger.warning("Request attempt %d to %s failed: %s. Retrying in %ds...", attempt, url, exc, delay)
                time.sleep(delay)
            else:
                logger.error("Request to %s failed after %d attempts: %s", url, retries, exc)
    
    raise last_exc or RuntimeError(f"Request to {url} failed unexpectedly")


# ---------------------------------------------------------------------------
# Web Search + Page Text Extraction (Brave, DuckDuckGo, Bing via Webshare proxy)
# ---------------------------------------------------------------------------

GOOGLE_SEARCH_OUTPUT_DIR = RESEARCH_ROOT / "google"
GOOGLE_SEARCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Domains to filter out from search results
SEARCH_FILTER_DOMAINS = [
    "youtube.com", "youtu.be",  # YouTube links (we get those separately)
    "google.com", "gstatic.com", "maps.google",
    "bing.com", "microsoft.com", "msn.com", "go.microsoft",
    "duckduckgo.com",
    "brave.com", "search.brave",
    "reddit.com", "redd.it",  # Reddit - low quality discussions
]


def _get_us_proxy() -> dict:
    """Get Webshare proxy configured for US country."""
    # Use US-specific proxy URL with -US-rotate username format
    return REQUESTS_PROXY_US


def _is_filtered_domain(url: str) -> bool:
    """Check if URL should be filtered out."""
    return any(domain in url.lower() for domain in SEARCH_FILTER_DOMAINS)


def _search_duckduckgo(query: str, num_results: int = 4) -> List[str]:
    """Search DuckDuckGo HTML version and return result URLs."""
    logger.info("Searching DuckDuckGo for: %s", query)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    us_proxy = _get_us_proxy()
    
    try:
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query, "kl": "us-en"},  # Force US English
            headers=headers,
            proxies=us_proxy,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
        return []
    
    soup = BeautifulSoup(response.text, "lxml")
    urls = []
    
    # DuckDuckGo wraps URLs in redirect: //duckduckgo.com/l/?uddg=<encoded_url>
    for a_tag in soup.select("a.result__a"):
        href = a_tag.get("href", "")
        
        # Extract actual URL from DuckDuckGo redirect
        if "uddg=" in href:
            from urllib.parse import unquote
            parsed = parse_qs(urlparse(href).query)
            if "uddg" in parsed:
                href = unquote(parsed["uddg"][0])
        
        if href.startswith("http") and not _is_filtered_domain(href) and href not in urls:
            urls.append(href)
            if len(urls) >= num_results:
                break
    
    logger.info("DuckDuckGo found %d URLs", len(urls))
    return urls


def _search_bing(query: str, num_results: int = 4) -> List[str]:
    """Search Bing using Playwright (JS-rendered) and return result URLs."""
    logger.info("Searching Bing for: %s", query)
    
    try:
        from playwright.sync_api import sync_playwright
        import base64
    except ImportError:
        logger.warning("Playwright not installed, skipping Bing search")
        return []
    
    # Build proxy config for Playwright (use US proxy)
    parsed = urlparse(WEBSHARE_PROXY_URL_US)
    proxy_config = {
        "server": f"http://{parsed.hostname}:{parsed.port}",
        "username": parsed.username,
        "password": parsed.password,
    }
    
    urls = []
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, proxy=proxy_config)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                locale="en-US",
                timezone_id="America/New_York",
            )
            page = context.new_page()
            
            # Navigate to Bing search - use mkt=en-US to force US market results
            search_url = f"https://www.bing.com/search?q={requests.utils.quote(query)}&mkt=en-US"
            page.goto(search_url, timeout=30000)
            
            # Wait for results to render (wait for any result item)
            page.wait_for_selector("li.b_algo", state="attached", timeout=10000)
            
            # Give extra time for links to fully render
            page.wait_for_timeout(1000)
            
            # Extract result links - Bing wraps real URLs in /ck/a? redirect
            results = page.query_selector_all("li.b_algo h2 a")
            for result in results:
                href = result.get_attribute("href") or ""
                
                # Bing uses tracking redirect URLs like /ck/a?...&u=a1<base64_url>&...
                if "/ck/a?" in href:
                    # Parse query params to get the 'u' parameter
                    qs = parse_qs(urlparse(href).query)
                    if "u" in qs:
                        u_param = qs["u"][0]
                        # Strip 'a1' prefix and base64 decode
                        if u_param.startswith("a1"):
                            try:
                                real_url = base64.urlsafe_b64decode(u_param[2:] + "==").decode("utf-8")
                                if real_url.startswith("http") and not _is_filtered_domain(real_url) and real_url not in urls:
                                    urls.append(real_url)
                            except Exception:
                                pass
                elif href.startswith("http") and not _is_filtered_domain(href) and href not in urls:
                    urls.append(href)
                
                if len(urls) >= num_results:
                    break
            
            browser.close()
            
    except Exception as exc:
        logger.warning("Bing Playwright search failed: %s", exc)
        return []
    
    logger.info("Bing found %d URLs", len(urls))
    return urls


def _search_brave(query: str, num_results: int = 4) -> List[str]:
    """Search Brave and return result URLs."""
    logger.info("Searching Brave for: %s", query)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    us_proxy = _get_us_proxy()
    
    try:
        response = requests.get(
            "https://search.brave.com/search",
            params={"q": query, "source": "web"},
            headers=headers,
            proxies=us_proxy,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.warning("Brave search failed: %s", exc)
        return []
    
    soup = BeautifulSoup(response.text, "lxml")
    urls = []
    
    # Brave results are in <a class="result-header"> or data-url attributes
    for a_tag in soup.select("a.result-header, div.snippet a[href], a[data-url]"):
        href = a_tag.get("href") or a_tag.get("data-url") or ""
        
        if href.startswith("http") and not _is_filtered_domain(href) and href not in urls:
            urls.append(href)
            if len(urls) >= num_results:
                break
    
    logger.info("Brave found %d URLs", len(urls))
    return urls


def multi_search_via_proxy(query: str, num_per_engine: int = 4) -> List[str]:
    """Search Brave, DuckDuckGo, and Bing. Return deduplicated URLs (max 4 per engine).
    
    Args:
        query: Search query string
        num_per_engine: Number of results per search engine (default 4)
        
    Returns:
        List of unique URLs from all search engines (duplicates removed)
    """
    logger.info("Performing multi-engine search for: %s", query)
    
    seen_urls = set()
    unique_urls = []
    
    # Search each engine and collect top 4 from each
    for engine_name, search_func in [
        ("Brave", _search_brave),
        ("DuckDuckGo", _search_duckduckgo),
        ("Bing", _search_bing),
    ]:
        try:
            engine_urls = search_func(query, num_per_engine)
            added = 0
            for url in engine_urls[:num_per_engine]:
                if url not in seen_urls:
                    seen_urls.add(url)
                    unique_urls.append(url)
                    added += 1
            logger.info("%s contributed %d unique URLs", engine_name, added)
        except Exception as exc:
            logger.warning("%s search error: %s", engine_name, exc)
    
    logger.info("Multi-search found %d unique URLs total", len(unique_urls))
    return unique_urls


# Keep backward compatibility alias
def google_search_via_proxy(query: str, num_results: int = 4) -> List[str]:
    """Alias for multi_search_via_proxy (Google blocked, using alternatives)."""
    return multi_search_via_proxy(query, num_per_engine=num_results)


def extract_page_text(url: str, max_chars: int = 10000) -> str:
    """Extract main text content from a webpage via Webshare proxy.
    
    Args:
        url: URL to fetch and extract text from
        max_chars: Maximum characters to extract (default 10000)
        
    Returns:
        Extracted text content, or empty string on error
    """
    logger.info("Extracting text from: %s", url)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    try:
        response = requests.get(
            url,
            headers=headers,
            proxies=REQUESTS_PROXY,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return ""
    
    soup = BeautifulSoup(response.content, "lxml")
    
    # Remove script, style, nav, footer, header elements
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        element.decompose()
    
    # Try to find main content areas first
    main_content = None
    for selector in ["article", "main", '[role="main"]', ".content", "#content", ".post-content", ".article-body"]:
        main_content = soup.select_one(selector)
        if main_content:
            break
    
    if main_content:
        text = main_content.get_text(separator="\n", strip=True)
    else:
        # Fall back to body text
        body = soup.find("body")
        text = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)
    
    # Clean up multiple whitespace/newlines
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[... truncated ...]"
        
    logger.info("Extracted %d characters from %s", len(text), url)
    return text


def fetch_google_research(topic: str, num_links: int = 4) -> Tuple[str, Path]:
    """Search Google for topic, extract text from top results, and merge into a text file.
    
    Args:
        topic: Topic to search for
        num_links: Number of result links to process (default 4)
        
    Returns:
        Tuple of (merged text content, path where it was saved)
    """
    logger.info("Starting Google research for topic: %s", topic)
    
    # Get search result URLs
    urls = google_search_via_proxy(topic, num_results=num_links)
    
    if not urls:
        logger.warning("No URLs found from Google search")
        return "", Path()
    
    # Extract text from each URL
    all_texts = []
    for i, url in enumerate(urls, 1):
        text = extract_page_text(url)
        if text:
            all_texts.append(f"=== SOURCE {i}: {url} ===\n\n{text}\n")
    
    if not all_texts:
        logger.warning("Failed to extract text from any URLs")
        return "", Path()
    
    # Merge all texts
    merged_text = "\n".join(all_texts)
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[^\w\s-]', '', topic)[:50].strip().replace(' ', '_')
    output_file = GOOGLE_SEARCH_OUTPUT_DIR / f"google_research_{safe_topic}_{timestamp}.txt"
    
    output_file.write_text(merged_text, encoding="utf-8")
    logger.info("Google research saved to: %s (%d chars from %d sources)", output_file, len(merged_text), len(all_texts))
    
    return merged_text, output_file


def run_tavily_research(topic: str, notes: str) -> Tuple[str, Path]:
    """Query Tavily API for research on the topic using provided notes as context.
    
    Returns the research answer text and the path where it was saved.
    """
    if not TAVILY_API_KEY:
        raise SystemExit(
            "TAVILY_API_KEY is not set.\n"
            "Get your API key from https://tavily.com and set it:\n"
            "  Windows:  setx TAVILY_API_KEY \"tvly-...\""
        )

    # Tavily query must be concise (under 400 chars) - just the topic search
    # The full notes will be combined with Tavily results for Gemini later
    search_query = f"{topic} analysis research insights trends"
    
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": search_query,
        "search_depth": "advanced",
        "include_answer": True,
        "include_raw_content": False,  # Reduces response size
        "max_results": 10,
    }

    logger.info("Submitting Tavily research request for '%s'", topic)
    
    try:
        response = _request_with_proxy(TAVILY_ENDPOINT, json=payload)
        data = response.json()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Tavily API request failed: {exc}") from exc
    
    # Extract answer from response
    answer = data.get("answer")
    if not answer:
        # Fallback: try to build answer from search results
        results = data.get("results", [])
        if results:
            answer = "\n\n".join(
                f"**{r.get('title', 'Untitled')}**\n{r.get('content', '')}"
                for r in results[:5]
            )
        else:
            logger.warning("Tavily returned no answer or results; saving raw response")
            answer = json.dumps(data, indent=2)

    slug = slugify(topic)
    timestamp = current_timestamp()
    answer_path = TAVILY_OUTPUT_DIR / f"{slug}_{timestamp}_tavily.txt"
    raw_path = TAVILY_OUTPUT_DIR / f"{slug}_{timestamp}_tavily.json"

    answer_path.write_text(answer, encoding="utf-8")
    raw_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Tavily research saved to %s (%d chars)", answer_path, len(answer))
    return answer, answer_path


def run_gemini_writer(topic: str, research_summary: str, notes: str) -> Tuple[str, Path]:
    """Send research and notes to Gemini to generate the documentary script.
    
    Returns the generated script text and the path where it was saved.
    """
    if not GEMINI_API_KEY:
        raise SystemExit(
            "GEMINI_API_KEY is not set.\n"
            "Get your API key from https://aistudio.google.com/apikey and set it:\n"
            "  Windows:  setx GEMINI_API_KEY \"AI...\""
        )

    if not GEMINI_MASTER_PROMPT.strip():
        raise SystemExit("GEMINI_MASTER_PROMPT is empty. Populate it with the desired instructions.")

    # Writer receives: research + YouTube transcripts + master prompt
    # This should be well under 200K TPM limit (~28K tokens typically)
    logger.info("Preparing writer prompt: research (%d chars) + YouTube transcripts (%d chars)", len(research_summary), len(notes))

    prompt = "\n\n".join(
        [
            GEMINI_MASTER_PROMPT.strip(),
            f"TOPIC: {topic}",
            "="*60,
            "RESEARCH ANALYSIS (use this as your primary guide):",
            "="*60,
            research_summary.strip(),
            "="*60,
            "YOUTUBE TRANSCRIPTS (for quotes, tone, and attribution):",
            "="*60,
            notes.strip(),
        ]
    )

    endpoint = GEMINI_API_URL_TEMPLATE.format(model=GEMINI_MODEL)
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 16384,  # Allow longer script output (~4000 words)
        }
    }

    logger.info("Sending research to Gemini model '%s' (prompt: %d chars)", GEMINI_MODEL, len(prompt))
    
    try:
        response = _request_with_retry(endpoint, params={"key": GEMINI_API_KEY}, json=payload)
        data = response.json()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Gemini API request failed: {exc}") from exc

    # Extract generated text
    try:
        candidates = data.get("candidates", [])
        if not candidates:
            error_info = data.get("error", {})
            if error_info:
                raise RuntimeError(f"Gemini API error: {error_info.get('message', json.dumps(error_info))}")
            raise RuntimeError(f"No candidates in Gemini response: {json.dumps(data, indent=2)}")
        
        text = candidates[0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Gemini response structure: {json.dumps(data, indent=2)}") from exc

    if not text:
        raise RuntimeError("Gemini returned an empty response")
    
    # Check word count and trim if needed
    word_count = len(text.split())
    logger.info("First pass: %d words", word_count)
    
    MAX_WORDS = 2500
    MAX_TRIM_ATTEMPTS = 2
    
    for attempt in range(1, MAX_TRIM_ATTEMPTS + 1):
        if word_count <= MAX_WORDS:
            break
        
        logger.warning("Script exceeds %d words (%d). Trim attempt %d...", MAX_WORDS, word_count, attempt)
        time.sleep(15)  # Rate limit
        
        trim_prompt = f"""The following script is {word_count} words, which EXCEEDS the 2,500 word limit.

You MUST cut it to 2,000-2,500 words. This is MANDATORY.

SMART TRIM PROTOCOL:
1. ANALYZE: Identify the core thesis (1 sentence) and essential evidence (max 3 key points)
2. PRIORITIZE: Rank examples by impact - keep only the strongest proof for each point
3. EXECUTE: Cut entire paragraphs that repeat earlier points or add tangential context

RULES:
- Cut whole paragraphs, not just sentences
- If a concept is explained twice, DELETE the weaker explanation entirely
- Remove "nice to have" context that doesn't prove the thesis
- Keep the hook, thesis, and conclusion intact

Output ONLY the trimmed script. No commentary.

SCRIPT TO TRIM:
{text}"""
        
        trim_payload = {
            "contents": [{"parts": [{"text": trim_prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 65536}
        }
        
        try:
            trim_response = _request_with_retry(endpoint, params={"key": GEMINI_API_KEY}, json=trim_payload)
            trim_data = trim_response.json()
            text = trim_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            word_count = len(text.split())
            logger.info("Trim attempt %d: %d words", attempt, word_count)
        except Exception as exc:
            logger.warning("Trim attempt %d failed: %s", attempt, exc)
            break
    
    if word_count > MAX_WORDS:
        logger.warning("⚠️ Script still exceeds limit (%d words). Manual trim needed.", word_count)
    else:
        logger.info("✓ Word count in target range (2,000-2,500)")

    slug = slugify(topic)
    timestamp = current_timestamp()
    script_path = GEMINI_OUTPUT_DIR / f"{slug}_{timestamp}_gemini.txt"
    script_path.write_text(text, encoding="utf-8")
    
    logger.info("Gemini script saved to %s (%d chars, ~%d words)", script_path, len(text), word_count)
    return text, script_path


def translate_script_to_spanish(english_script: str, output_path: Path) -> Tuple[str, Path]:
    """Translate an English script to neutral Latin American Spanish using Gemini.
    
    Args:
        english_script: The English script text to translate
        output_path: Path where the English script was saved (used to derive Spanish path)
        
    Returns:
        Tuple of (translated Spanish script, path where it was saved)
    """
    if not GEMINI_API_KEY:
        raise SystemExit("GEMINI_API_KEY is not set.")
    
    prompt = GEMINI_TRANSLATION_PROMPT.format(script=english_script)
    
    endpoint = GEMINI_API_URL_TEMPLATE.format(model=GEMINI_MODEL)
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,  # Lower temp for more faithful translation
            "maxOutputTokens": 65536,  # Increased from 16384 to avoid truncation
        }
    }
    
    english_word_count = len(english_script.split())
    logger.info("Translating script to Spanish (%d chars, ~%d words)...", len(english_script), english_word_count)
    
    try:
        response = _request_with_retry(endpoint, params={"key": GEMINI_API_KEY}, json=payload)
        data = response.json()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Gemini translation API request failed: {exc}") from exc
    
    # Extract translated text
    try:
        candidates = data.get("candidates", [])
        if not candidates:
            error_info = data.get("error", {})
            if error_info:
                raise RuntimeError(f"Gemini API error: {error_info.get('message', json.dumps(error_info))}")
            raise RuntimeError(f"No candidates in Gemini response: {json.dumps(data, indent=2)}")
        
        spanish_text = candidates[0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Gemini response structure: {json.dumps(data, indent=2)}") from exc
    
    if not spanish_text:
        raise RuntimeError("Gemini returned an empty translation")
    
    # Check for potential truncation (Spanish should be ~90-110% of English word count)
    spanish_word_count = len(spanish_text.split())
    english_word_count = len(english_script.split())
    ratio = spanish_word_count / english_word_count if english_word_count > 0 else 0
    
    if ratio < 0.85:
        logger.warning("⚠️  POSSIBLE TRUNCATION: Spanish (%d words) is only %.0f%% of English (%d words)", 
                       spanish_word_count, ratio * 100, english_word_count)
        logger.warning("   The translation may be incomplete. Check the output file.")
    
    # Save Spanish version with _es suffix
    spanish_path = output_path.with_name(output_path.stem + "_es.txt")
    spanish_path.write_text(spanish_text, encoding="utf-8")
    
    word_count = len(spanish_text.split())
    logger.info("Spanish script saved to %s (%d chars, ~%d words)", spanish_path, len(spanish_text), word_count)
    return spanish_text, spanish_path


def run_gemini_research(topic: str, notes: str) -> Tuple[str, Path]:
    """Use Gemini to deeply analyze source notes and produce structured research.
    
    This replaces Tavily for research - Gemini analyzes all collected notes
    (YouTube transcripts, web articles) and produces a comprehensive research document.
    
    Returns the research text and path where it was saved.
    """
    if not GEMINI_API_KEY:
        raise SystemExit(
            "GEMINI_API_KEY is not set.\n"
            "Get your API key from https://aistudio.google.com/apikey and set it:\n"
            "  Windows:  setx GEMINI_API_KEY \"AI...\""
        )

    # Gemini 2.5 Pro has ~1M token context - use full notes without truncation
    logger.info("Processing %d chars of source notes for research", len(notes))

    prompt = GEMINI_RESEARCH_PROMPT.format(topic=topic, notes=notes)

    endpoint = GEMINI_API_URL_TEMPLATE.format(model=GEMINI_RESEARCH_MODEL)
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,  # Lower temp for factual research
            "maxOutputTokens": 16384,  # Allow longer research output
        }
    }

    logger.info("Sending to Gemini research model '%s' (prompt: %d chars)", GEMINI_RESEARCH_MODEL, len(prompt))
    
    try:
        response = _request_with_retry(endpoint, params={"key": GEMINI_API_KEY}, json=payload)
        data = response.json()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Gemini research API request failed: {exc}") from exc

    # Extract generated text
    try:
        candidates = data.get("candidates", [])
        if not candidates:
            error_info = data.get("error", {})
            if error_info:
                raise RuntimeError(f"Gemini API error: {error_info.get('message', json.dumps(error_info))}")
            raise RuntimeError(f"No candidates in Gemini response: {json.dumps(data, indent=2)}")
        
        text = candidates[0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Gemini response structure: {json.dumps(data, indent=2)}") from exc

    if not text:
        raise RuntimeError("Gemini research returned an empty response")

    slug = slugify(topic)
    timestamp = current_timestamp()
    research_path = GEMINI_OUTPUT_DIR / f"{slug}_{timestamp}_research.txt"
    research_path.write_text(text, encoding="utf-8")
    
    logger.info("Gemini research saved to %s (%d chars)", research_path, len(text))
    return text, research_path


# KIMI RESEARCH FUNCTION - DISABLED
# Kimi API was unreliable, keeping Gemini-only for now
# To re-enable: uncomment this function and the KIMI_* constants above
#
# def run_kimi_research(topic: str, notes: str) -> Tuple[str, Path]:
#     """Use Kimi (Moonshot AI) to analyze source notes and produce research.
#     
#     Kimi has 128k context and is good for long-form analysis.
#     Respects free tier rate limits: ~3 requests/min, 1 concurrent.
#     
#     Returns the research text and path where it was saved.
#     """
#     if not KIMI_API_KEY:
#         logger.warning("KIMI_API_KEY not set, skipping Kimi research")
#         return "", Path()
#     ... (rest of function commented out)


def run_research_pipeline(topic: str, transcripts: List[TranscriptResult], extra_notes: str = "") -> Tuple[Path, Path]:
    """Run the full research pipeline: Gemini research -> Gemini writing.
    
    INPUTS TO RESEARCH (Gemini):
    - Search engine scraped content (extra_notes) + YouTube transcripts
    - Research prompt with analysis instructions
    
    INPUTS TO WRITING (Gemini):
    - Research report from Gemini
    - YouTube transcripts only (for quotes/attribution)
    - Writing prompt (GEMINI_MASTER_PROMPT)
    """
    # Log input sources
    yt_char_count = sum(len(t.text) for t in transcripts) if transcripts else 0
    search_char_count = len(extra_notes) if extra_notes else 0
    logger.info("="*60)
    logger.info("RESEARCH PIPELINE INPUT SOURCES:")
    logger.info("  - YouTube transcripts: %d videos, %d chars", len(transcripts), yt_char_count)
    logger.info("  - Search engine content: %d chars", search_char_count)
    logger.info("="*60)
    
    # Compose full notes for research (web + YouTube)
    all_notes = compose_research_notes(transcripts, extra_notes)
    if not all_notes.strip():
        raise SystemExit("Cannot run research pipeline without any notes.")
    
    logger.info("Combined source notes for research: %d chars total", len(all_notes))

    # Step 1: Gemini deep research (uses ALL sources)
    logger.info("Step 1/2: Running Gemini deep research analysis...")
    gemini_research, gemini_path = run_gemini_research(topic, all_notes)
    
    # Wait for rate limit before next Gemini request (2 RPM = 30s between requests)
    logger.info("Waiting %d seconds for Gemini rate limit (2 RPM)...", GEMINI_RATE_LIMIT_DELAY)
    time.sleep(GEMINI_RATE_LIMIT_DELAY)
    
    # Build YouTube-only notes for writer (for quotes/attribution)
    youtube_notes = build_notes_from_transcripts(transcripts)
    logger.info("YouTube transcripts for writer: %d chars", len(youtube_notes))
    
    # Step 2: Gemini script writing (research + YouTube transcripts only)
    logger.info("Step 2/3: Running Gemini script writing (English)...")
    script_text, script_path = run_gemini_writer(topic, gemini_research, youtube_notes)
    
    # Wait for rate limit before translation
    logger.info("Waiting %d seconds for Gemini rate limit before translation...", GEMINI_RATE_LIMIT_DELAY)
    time.sleep(GEMINI_RATE_LIMIT_DELAY)
    
    # Step 3: Translate to Spanish
    logger.info("Step 3/3: Translating script to Spanish...")
    spanish_text, spanish_path = translate_script_to_spanish(script_text, script_path)
    
    logger.info("Research pipeline complete for '%s'", topic)
    logger.info("  English script: %s", script_path)
    logger.info("  Spanish script: %s", spanish_path)
    return gemini_path, script_path


def extract_video_id(url: str) -> Optional[str]:
    """Extract the video ID from a YouTube URL.
    
    Supported formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/v/VIDEO_ID
    """
    parsed = urlparse(url)
    netloc = parsed.netloc.lower().replace("www.", "")
    
    if netloc == "youtu.be":
        return parsed.path.strip("/").split("?")[0] or None
    
    if netloc in {"youtube.com", "m.youtube.com"}:
        path = parsed.path
        if path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        for prefix in ("/shorts/", "/embed/", "/v/"):
            if path.startswith(prefix):
                return path[len(prefix):].split("/")[0].split("?")[0] or None
    
    return None


def _parse_webshare_credentials() -> Tuple[str, str]:
    """Extract username and password from WEBSHARE_PROXY_URL.
    
    Expected format: http://username:password@host:port
    Returns (username, password) tuple.
    """
    if not WEBSHARE_PROXY_URL:
        raise ValueError("WEBSHARE_PROXY_URL is not set")
    
    parsed = urlparse(WEBSHARE_PROXY_URL)
    username = parsed.username or ""
    password = parsed.password or ""
    
    if not username or not password:
        raise ValueError(f"Could not extract credentials from proxy URL: {WEBSHARE_PROXY_URL}")
    
    return username, password


def fetch_captions(video_id: str) -> Optional[str]:
    """Return transcript text via embedded captions if available.
    
    Includes retry logic for transient network failures.
    Uses youtube-transcript-api v1.x instance-based API with Webshare proxy.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Parse Webshare credentials from proxy URL
            proxy_user, proxy_pass = _parse_webshare_credentials()
            
            # Instantiate API with Webshare proxy config per call for thread-safety
            api = YouTubeTranscriptApi(
                proxy_config=WebshareProxyConfig(
                    proxy_username=proxy_user,
                    proxy_password=proxy_pass,
                )
            )
            transcript = api.fetch(video_id, languages=PREFERRED_LANGUAGES)
            break
        except (TranscriptsDisabled, NoTranscriptFound):
            return None
        except Exception as exc:  # noqa: BLE001
            if attempt < MAX_RETRIES:
                logger.warning("Caption fetch attempt %d failed for %s: %s. Retrying...", attempt, video_id, exc)
                time.sleep(RETRY_DELAY_SECONDS * attempt)
            else:
                logger.exception("Unexpected error while fetching captions for %s after %d attempts", video_id, MAX_RETRIES)
                return None
    else:
        return None

    # Clean and join transcript chunks, preserving sentence boundaries
    # New API returns FetchedTranscript with FetchedTranscriptSnippet objects (not dicts)
    texts = [snippet.text.strip() for snippet in transcript if snippet.text and snippet.text.strip()]
    return " ".join(texts)


def fetch_video_metadata(url: str) -> dict:
    """Return yt_dlp metadata without downloading.
    
    Includes retry logic for transient network failures.
    """
    ydl_opts = _build_ydl_opts()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=False)
        except Exception as exc:  # noqa: BLE001 - metadata is optional
            if attempt < MAX_RETRIES:
                logger.warning("Metadata fetch attempt %d failed for %s: %s. Retrying...", attempt, url, exc)
                time.sleep(RETRY_DELAY_SECONDS * attempt)
            else:
                logger.warning("Could not fetch metadata for %s after %d attempts", url, MAX_RETRIES)
    return {}


def search_youtube(topic: str, limit: int = YOUTUBE_RESULT_LIMIT_DEFAULT) -> List[str]:
    """Return the top YouTube video URLs for the given topic using yt_dlp search.
    
    Includes retry logic for transient network failures.
    """
    limit = max(1, min(limit, 50))  # Cap at 50 to avoid excessive results
    query = f"ytsearch{limit}:{topic}"
    ydl_opts = _build_ydl_opts()
    
    info = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(query, download=False)
            break
        except Exception as exc:
            if attempt < MAX_RETRIES:
                logger.warning("YouTube search attempt %d failed for '%s': %s. Retrying...", attempt, topic, exc)
                time.sleep(RETRY_DELAY_SECONDS * attempt)
            else:
                logger.exception("YouTube search failed for topic '%s' after %d attempts", topic, MAX_RETRIES)
                return []
    
    if not info:
        return []

    entries = info.get("entries") or []
    urls: List[str] = []
    for entry in entries:
        video_id = entry.get("id")
        if video_id:
            urls.append(f"https://www.youtube.com/watch?v={video_id}")
        if len(urls) >= limit:
            break
    
    logger.info("Found %d video(s) for topic '%s'", len(urls), topic)
    return urls


def download_audio(url: str, target_dir: Path) -> Path:
    """Download audio track for Whisper processing and return the file path.
    
    Includes retry logic for transient network failures.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(target_dir / "%(id)s.%(ext)s")
    ydl_opts = _build_ydl_opts(
        format="bestaudio/best",
        outtmpl=outtmpl,
        skip_download=False,
        postprocessors=[
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    )
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded = Path(ydl.prepare_filename(info))
            break
        except Exception as exc:
            if attempt < MAX_RETRIES:
                logger.warning("Audio download attempt %d failed for %s: %s. Retrying...", attempt, url, exc)
                time.sleep(RETRY_DELAY_SECONDS * attempt)
            else:
                raise

    wav_path = downloaded.with_suffix(".wav")
    return wav_path if wav_path.exists() else downloaded


def load_whisper_model():  # type: ignore[return-any]
    """Load Whisper base on first use."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        # Lazy import torch to avoid startup failure when not using Whisper
        try:
            import torch
        except ImportError as exc:
            raise SystemExit(
                "torch is required for Whisper fallback transcriptions.\n"
                "Install with: pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113"
            ) from exc
        
        cuda_available = torch.cuda.is_available()
        if not cuda_available and not ALLOW_WHISPER_CPU:
            raise SystemExit(
                "A CUDA-capable GPU is required for Whisper fallback transcriptions.\n"
                "Set WHISPER_ALLOW_CPU=1 or --allow-whisper-cpu to allow slow CPU fallback."
            )
        try:
            import whisper  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise SystemExit("whisper is required. Install with 'pip install -U openai-whisper'.") from exc

        device = "cuda" if cuda_available else "cpu"
        logger.info("Loading Whisper base model on %s (one-time)...", device.upper())
        _WHISPER_MODEL = whisper.load_model("base", device=device)
    return _WHISPER_MODEL


def transcribe_with_whisper(url: str) -> Optional[str]:
    """Download audio and run Whisper if captions do not exist."""
    with tempfile.TemporaryDirectory(prefix="yt_whisper_", dir=str(WHISPER_TMP_DIR)) as tmp_dir:
        tmp_path = Path(tmp_dir)
        try:
            logger.info("Whisper fallback: downloading audio for %s", url)
            audio_path = download_audio(url, tmp_path)
        except Exception:
            logger.exception("Failed to download audio for %s", url)
            return None

        try:
            logger.info("Whisper fallback: transcribing %s", audio_path.name)
            model = load_whisper_model()
            result = model.transcribe(str(audio_path), fp16=False)
            return result.get("text", "").strip()
        except Exception:
            logger.exception("Whisper transcription failed for %s", url)
            return None


def process_video(url: str) -> Optional[TranscriptResult]:
    """Process a single video URL and return its transcript.
    
    Attempts embedded captions first, falls back to Whisper if unavailable.
    """
    video_id = extract_video_id(url)
    if not video_id:
        logger.error("Could not determine video id for %s", url)
        return None

    metadata = fetch_video_metadata(url)
    title = metadata.get("title") or video_id
    duration = metadata.get("duration", 0) or 0

    caption_text = fetch_captions(video_id)
    if caption_text:
        logger.info("Using embedded captions for '%s' (%d chars)", title, len(caption_text))
        return TranscriptResult(
            title=title,
            url=url,
            source="captions",
            text=caption_text,
            video_id=video_id,
            duration_seconds=duration,
        )

    logger.info("Captions unavailable for '%s'. Falling back to Whisper.", title)
    whisper_text = transcribe_with_whisper(url)
    if whisper_text:
        logger.info("Whisper transcription complete for '%s' (%d chars)", title, len(whisper_text))
        return TranscriptResult(
            title=title,
            url=url,
            source="whisper",
            text=whisper_text,
            video_id=video_id,
            duration_seconds=duration,
        )

    logger.error("Failed to obtain transcript for %s", url)
    return None


def cleanup_old_temp_files() -> int:
    """Remove temp files older than TEMP_CLEANUP_AGE_HOURS.
    
    Returns the number of files removed.
    """
    if not TEMP_DIR.exists():
        return 0
    
    cutoff_time = time.time() - (TEMP_CLEANUP_AGE_HOURS * 3600)
    removed = 0
    
    for item in TEMP_DIR.rglob("*"):
        if item.is_file():
            try:
                if item.stat().st_mtime < cutoff_time:
                    item.unlink()
                    removed += 1
            except OSError:
                pass  # File may be in use or already deleted
    
    if removed:
        logger.info("Cleaned up %d old temp file(s)", removed)
    return removed


def write_merged_transcripts(results: List[TranscriptResult], output_path: Path) -> None:
    """Write all transcripts to a single merged output file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_duration = sum(r.duration_seconds for r in results)
    caption_count = sum(1 for r in results if r.source == "captions")
    whisper_count = len(results) - caption_count
    
    with output_path.open("w", encoding="utf-8") as file:
        # Write header with summary
        file.write(f"# Merged YouTube Transcripts\n")
        file.write(f"Generated: {datetime.utcnow().isoformat()}Z\n")
        file.write(f"Videos: {len(results)} | Duration: {format_duration(total_duration)}\n")
        file.write(f"Sources: {caption_count} captions, {whisper_count} whisper\n")
        file.write("=" * 60 + "\n\n")
        
        for idx, result in enumerate(results, start=1):
            file.write(f"### Video {idx}: {result.title}\n")
            file.write(f"URL: {result.url}\n")
            file.write(f"Duration: {format_duration(result.duration_seconds)}\n")
            file.write(f"Source: {result.source}\n\n")
            file.write(result.text.strip())
            file.write("\n\n")

    logger.info("Merged transcript saved to %s (%d videos, %d chars)", 
                output_path, len(results), output_path.stat().st_size)


def _fetch_metadata_and_caption(url: str) -> Tuple[dict, Optional[str]]:
    """Fetch metadata and captions for a URL. Returns (metadata, caption_text or None)."""
    metadata = fetch_video_metadata(url) or {}
    video_id = extract_video_id(url)
    caption = None
    if video_id:
        try:
            caption = fetch_captions(video_id)
        except Exception:
            caption = None
    return metadata, caption


def validate_environment(require_whisper: bool = False, require_api: bool = False):
    """Validate runtime environment and raise SystemExit with helpful tips when misconfigured."""
    # ffmpeg is required for audio extraction
    if not FFMPEG_PATH:
        raise SystemExit(
            "ffmpeg is required for audio extraction. Install it and ensure it's in PATH.\n"
            "Windows (chocolatey): choco install ffmpeg\n"
            "Or: https://ffmpeg.org/download.html"
        )

    # Check I: drive
    if not I_DRIVE_ROOT.exists():
        raise SystemExit("Drive I: is required. Please mount it and re-run the script.")

    # Whisper GPU check if needed
    if require_whisper:
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            raise SystemExit(
                "torch is required for Whisper fallback transcriptions.\n"
                "Install with: pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113"
            )
        
        if not cuda_available:
            if ALLOW_WHISPER_CPU:
                logger.warning("CUDA is not available; Whisper will run on CPU which may be very slow.")
            else:
                raise SystemExit(
                    "A CUDA-capable GPU is required for Whisper fallback transcriptions. Set WHISPER_ALLOW_CPU=1 to allow CPU fallback at your own risk."
                )

    # Check API keys when research is requested
    if require_api:
        missing = []
        if not TAVILY_API_KEY:
            missing.append("TAVILY_API_KEY")
        if not GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        if missing:
            raise SystemExit(
                "Missing API keys: {}. Set them as environment variables before running research.".format(
                    ", ".join(missing)
                )
            )

    logger.info("Environment validated: ffmpeg in PATH, I: drive present, proxy configured.")


def merge_urls(urls: Iterable[str], output_path: Path = OUTPUT_PATH, concurrency: Optional[int] = None, whisper_workers: Optional[int] = None, allow_whisper_cpu: Optional[bool] = None) -> Tuple[List[TranscriptResult], int]:
    """Process multiple URLs and write merged transcripts.

    Returns a tuple of (successful transcripts, failed count) without exiting
    the process, so it can be reused programmatically.
    """
    # Cleanup old temp files before starting
    cleanup_old_temp_files()
    
    url_list = [u.strip() for u in urls if u.strip()]
    total = len(url_list)
    
    if not url_list:
        logger.error("No valid URLs provided.")
        return [], 0
    
    transcripts: List[TranscriptResult] = []
    failed = 0
    
    # Phase 1: Fetch captions + metadata concurrently
    concurrency = concurrency or DEFAULT_METADATA_CONCURRENCY
    whisper_workers = whisper_workers or DEFAULT_WHISPER_WORKERS
    allow_whisper_cpu = allow_whisper_cpu if allow_whisper_cpu is not None else ALLOW_WHISPER_CPU
    metadata_workers = min(concurrency, max(1, total))
    logger.info("Fetching metadata and captions with %d workers...", metadata_workers)
    pending_whisper = []
    with ThreadPoolExecutor(max_workers=metadata_workers) as pool:
        future_to_url = {pool.submit(_fetch_metadata_and_caption, url): url for url in url_list}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                metadata, caption = future.result()
            except Exception as exc:
                logger.warning("Fetch failed for %s: %s", url, exc)
                failed += 1
                continue

            video_id = extract_video_id(url)
            title = metadata.get("title") or url
            duration = metadata.get("duration", 0) or 0
            if caption:
                transcripts.append(TranscriptResult(title=title, url=url, source="captions", text=caption, video_id=video_id, duration_seconds=duration))
            else:
                pending_whisper.append((url, metadata, video_id, title, duration))

    # Phase 2: Transcribe remaining videos with Whisper (limited concurrency)
    if pending_whisper:
        whisper_workers = min(DEFAULT_WHISPER_WORKERS, max(1, len(pending_whisper)))
        if whisper_workers > 1:
            cuda_available = False
            try:
                import torch  # type: ignore
                cuda_available = torch.cuda.is_available()
            except ImportError:
                logger.warning("Multiple Whisper workers requested but torch is not installed; reducing to 1")
            if not cuda_available and not ALLOW_WHISPER_CPU:
                logger.warning("Whisper worker count >1 requested but CUDA not available; reducing to 1")
                whisper_workers = 1

        if whisper_workers > 1 and ALLOW_WHISPER_CPU and not cuda_available:
            logger.info("Running multiple Whisper workers on CPU; expect slower performance")

        logger.info("Transcribing %d video(s) with Whisper (%d worker(s))...", len(pending_whisper), whisper_workers)
        if whisper_workers == 1:
            for (url, metadata, video_id, title, duration) in pending_whisper:
                whisper_text = transcribe_with_whisper(url)
                if whisper_text:
                    transcripts.append(TranscriptResult(title=title, url=url, source="whisper", text=whisper_text, video_id=video_id, duration_seconds=duration))
                else:
                    failed += 1
        else:
            with ThreadPoolExecutor(max_workers=whisper_workers) as pool:
                futures = {pool.submit(transcribe_with_whisper, url): (url, metadata, video_id, title, duration) for (url, metadata, video_id, title, duration) in pending_whisper}
                for future in as_completed(futures):
                    url, metadata, video_id, title, duration = futures[future]
                    try:
                        whisper_text = future.result()
                    except Exception as exc:
                        logger.warning("Whisper error for %s: %s", url, exc)
                        failed += 1
                        continue
                    if whisper_text:
                        transcripts.append(TranscriptResult(title=title, url=url, source="whisper", text=whisper_text, video_id=video_id, duration_seconds=duration))
                    else:
                        failed += 1

    if not transcripts:
        logger.error("No transcripts were generated. Check the logs above for details.")
        return [], failed

    write_merged_transcripts(transcripts, output_path)
    
    if failed:
        logger.warning("%d of %d video(s) failed to process", failed, total)
    
    return transcripts, failed


def main(urls: Iterable[str]) -> Tuple[List[TranscriptResult], int]:
    """Main entry point for programmatic use."""
    return merge_urls(urls, OUTPUT_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge transcripts for top YouTube results on a topic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s "machine learning basics" --limit 10
  %(prog)s --urls https://youtube.com/watch?v=abc123 https://youtu.be/xyz789
  %(prog)s --cleanup-only
""",
    )
    parser.add_argument(
        "topic",
        nargs="?",
        help="Topic keyword to search on YouTube when URLs are not provided.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=YOUTUBE_RESULT_LIMIT_DEFAULT,
        help=f"Number of top YouTube videos to include (default: {YOUTUBE_RESULT_LIMIT_DEFAULT}, max: 50).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Destination file for the merged transcript (default: {OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        metavar="URL",
        help="Explicit list of YouTube video URLs to merge instead of searching.",
    )
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Only clean up old temp files without processing any videos.",
    )
    parser.add_argument(
        "--run-research",
        action="store_true",
        help="After merging transcripts, send the consolidated notes to Tavily and Gemini.",
    )
    parser.add_argument(
        "--translate",
        type=Path,
        metavar="FILE",
        help="Translate an existing English script (.txt) to neutral Latin American Spanish. Outputs FILE_es.txt.",
    )
    parser.add_argument(
        "--notes-file",
        type=Path,
        help="Optional path to supplemental notes (e.g., Google summaries) to prepend before research.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_METADATA_CONCURRENCY,
        help=f"Number of workers for metadata/caption fetching (default: {DEFAULT_METADATA_CONCURRENCY}).",
    )
    parser.add_argument(
        "--whisper-workers",
        type=int,
        default=DEFAULT_WHISPER_WORKERS,
        help=f"Number of concurrent whisper workers (default: {DEFAULT_WHISPER_WORKERS}). Use 1 for GPU safety.",
    )
    parser.add_argument(
        "--allow-whisper-cpu",
        action="store_true",
        help="Allow Whisper CPU fallback if CUDA is not available (slow).",
    )
    parser.add_argument(
        "--google-links",
        type=int,
        default=0,
        metavar="N",
        help="Number of Google search result pages to scrape and include in research notes (default: 0, recommended: 4).",
    )
    return parser.parse_args()


def resolve_urls(args: argparse.Namespace) -> List[str]:
    if args.urls:
        if args.run_research and not args.topic:
            args.topic = ensure_topic_label(args)
        return args.urls

    topic = args.topic or input("Enter a topic to search on YouTube: ").strip()
    if not topic:
        raise SystemExit("A topic keyword is required when URLs are not supplied.")
    args.topic = topic

    urls = search_youtube(topic, args.limit)
    if not urls:
        raise SystemExit(f"No usable YouTube results were found for topic '{topic}'.")

    logger.info("Collected %d video(s) for topic '%s'.", len(urls), topic)
    return urls


def ensure_topic_label(args: argparse.Namespace) -> str:
    topic = getattr(args, "topic", None)
    if topic:
        return topic
    topic = input("Enter a topic label for the research prompts: ").strip()
    if not topic:
        raise SystemExit("A topic label is required when --run-research is enabled.")
    args.topic = topic
    return topic


def cli_main() -> None:
    args = parse_args()
    log_environment_summary(args.run_research)
    
    # Handle standalone translation mode
    if args.translate:
        if not args.translate.exists():
            raise SystemExit(f"Translation source file not found: {args.translate}")
        if not GEMINI_API_KEY:
            raise SystemExit("GEMINI_API_KEY is required for translation.")
        logger.info("="*60)
        logger.info("STANDALONE TRANSLATION MODE")
        logger.info("  Source: %s", args.translate)
        logger.info("  Rate limit: %d seconds between Gemini calls", GEMINI_RATE_LIMIT_DELAY)
        logger.info("="*60)
        english_script = args.translate.read_text(encoding="utf-8")
        word_count = len(english_script.split())
        logger.info("Loaded English script: %d chars, ~%d words", len(english_script), word_count)
        spanish_text, spanish_path = translate_script_to_spanish(english_script, args.translate)
        logger.info("="*60)
        logger.info("TRANSLATION COMPLETE")
        logger.info("  Output: %s", spanish_path)
        logger.info("="*60)
        return
    
    if args.cleanup_only:
        removed = cleanup_old_temp_files()
        logger.info("Cleanup complete. Removed %d file(s).", removed)
        return
    
    urls = resolve_urls(args)
    # Environment validation
    validate_environment(require_whisper=(args.whisper_workers > 0 or args.run_research), require_api=args.run_research)

    transcripts, failed = merge_urls(urls, args.output, concurrency=args.concurrency, whisper_workers=args.whisper_workers, allow_whisper_cpu=args.allow_whisper_cpu)

    if args.run_research and transcripts:
        topic_label = ensure_topic_label(args)
        extra_notes = read_additional_notes(args.notes_file)
        
        # Step 2: Google search + page extraction (if enabled)
        google_research_text = ""
        if args.google_links > 0:
            logger.info("Starting Google research with %d links...", args.google_links)
            google_research_text, google_output_path = fetch_google_research(topic_label, num_links=args.google_links)
            if google_research_text:
                logger.info("Google research complete: %d chars from %s", len(google_research_text), google_output_path)
            else:
                logger.warning("Google research returned no content")
        
        # Combine search engine content: Google research + additional notes file
        # YouTube transcripts are passed separately and combined in run_research_pipeline
        search_engine_notes = ""
        if google_research_text:
            search_engine_notes += f"{'='*60}\n=== SEARCH ENGINE SCRAPED ARTICLES ===\n{'='*60}\n\n{google_research_text}\n\n"
        if extra_notes:
            search_engine_notes += f"{'='*60}\n=== ADDITIONAL NOTES FILE ===\n{'='*60}\n\n{extra_notes}\n\n"
        
        logger.info("="*60)
        logger.info("PIPELINE STEP 3: Running LLM Research + Writing")
        logger.info("  Input: %d YouTube transcripts", len(transcripts))
        logger.info("  Input: %d chars search engine content", len(search_engine_notes))
        logger.info("="*60)
        
        run_research_pipeline(topic_label, transcripts, search_engine_notes)

    # Exit with non-zero code if all videos failed or any failures occurred
    if not transcripts or failed:
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
