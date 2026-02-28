#!/usr/bin/env python3
"""YouTube Description Generator (Gemini-Powered)

Generates SEO-optimized YouTube video descriptions using Gemini 2.5 Pro.

Usage:
    python youtube_description.py script.txt --keyword "crypto exchanges" --title "Best Crypto Exchanges 2025" --chapters "0:00 Intro,1:30 Top Picks,5:00 Fees"

Inputs:
    1. Script file (.txt) - The video script content
    2. --keyword - Primary SEO keyword (appears in first 2 sentences)
    3. --title - YouTube video title (reinforced in description)
    4. --chapters - Comma-separated chapter list (included verbatim)

Output:
    Formatted YouTube description with:
    - Hook (first 2 lines visible in preview)
    - Key points from script
    - Chapters (unchanged)
    - Call to action
    - Hashtags (auto-generated)

Requires:
    - GEMINI_API_KEY environment variable or .env file
    - pip install requests python-dotenv
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Gemini API config (same as youtube_transcript_merger.py)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Rate limit handling (same as youtube_transcript_merger.py)
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2.0
GEMINI_RATE_LIMIT_DELAY = 35  # 2 RPM = 1 request per 30 seconds, use 35 to be safe
REQUEST_TIMEOUT_SECONDS = 300  # 5 minutes for large prompts


def _request_with_retry(url: str, retries: int = MAX_RETRIES, **kwargs) -> requests.Response:
    """Make an HTTP POST request with retry logic for rate limits."""
    kwargs.setdefault("timeout", REQUEST_TIMEOUT_SECONDS)
    
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(url, **kwargs)
            
            # Handle 429 rate limit (Gemini: 2 RPM limit)
            if response.status_code == 429:
                wait_time = GEMINI_RATE_LIMIT_DELAY * attempt
                print(f"⏳ Rate limited (429). Waiting {wait_time}s before retry...", file=sys.stderr)
                time.sleep(wait_time)
                if attempt < retries:
                    continue
                    
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < retries:
                delay = RETRY_DELAY_SECONDS * attempt
                if "429" in str(exc) or "Too Many Requests" in str(exc):
                    delay = GEMINI_RATE_LIMIT_DELAY * attempt
                print(f"⚠️ Attempt {attempt} failed: {exc}. Retrying in {delay}s...", file=sys.stderr)
                time.sleep(delay)
            else:
                print(f"❌ Request failed after {retries} attempts: {exc}", file=sys.stderr)
    
    raise last_exc or RuntimeError(f"Request to {url} failed unexpectedly")


def generate_description_with_gemini(
    script_text: str,
    keyword: str,
    title: str,
    chapters: str,
) -> str:
    """Use Gemini 2.5 Pro to generate an SEO-optimized YouTube description."""
    
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables or .env file")
    
    # Build the prompt
    prompt = f"""You are a YouTube SEO expert. Generate a highly optimized YouTube video description.

INPUTS:
- Video Title: {title}
- Primary SEO Keyword: {keyword}
- Chapters (INCLUDE EXACTLY AS PROVIDED, DO NOT MODIFY):
{chapters}

- Video Script (for context):
{script_text[:8000]}

REQUIREMENTS:
1. HOOK (First 2 lines - CRITICAL for CTR):
   - Must include the keyword "{keyword}" in the first sentence
   - Make it compelling and curiosity-inducing
   - These lines appear in YouTube search results

2. DESCRIPTION BODY:
   - Summarize key value points from the script
   - Use bullet points (•) for readability
   - Include the keyword naturally 2-3 more times
   - Add relevant emojis sparingly

3. CHAPTERS:
   - Include the chapters EXACTLY as provided above
   - Format with ⏱️ Chapters: header
   - DO NOT modify timestamps or titles

4. CALL TO ACTION:
   - Subscribe, like, comment prompts
   - Bell notification reminder

5. HASHTAGS (at the end):
   - Generate 10-15 relevant hashtags
   - Include #{keyword.replace(' ', '')} as first hashtag
   - Mix broad and specific tags
   - Format: #Tag1 #Tag2 #Tag3

FORMAT:
- Use line breaks for readability
- Keep under 5000 characters total
- Do not use markdown formatting (no ** or ##)

Generate the description now:"""

    # Call Gemini via REST API (same as youtube_transcript_merger.py)
    endpoint = GEMINI_API_URL_TEMPLATE.format(model=GEMINI_MODEL)
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 4096,
        }
    }
    
    try:
        response = _request_with_retry(
            endpoint,
            params={"key": GEMINI_API_KEY},
            json=payload,
        )
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}", file=sys.stderr)
        raise
    
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
    
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Generate SEO-optimized YouTube video descriptions using Gemini 2.5 Pro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python youtube_description.py script.txt --keyword "crypto" --title "Best Crypto 2025"
  python youtube_description.py script.txt -k "python tutorial" -t "Learn Python" -c "0:00 Intro,2:00 Setup"
        """
    )
    parser.add_argument("script", type=Path, help="Path to the script .txt file")
    parser.add_argument("-k", "--keyword", required=True, help="Primary SEO keyword")
    parser.add_argument("-t", "--title", required=True, help="YouTube video title")
    parser.add_argument("-c", "--chapters", default="", help="Chapters (comma-separated, e.g., '0:00 Intro,1:30 Topic')")
    parser.add_argument("-o", "--output", type=Path, help="Output file (default: print to stdout)")
    
    args = parser.parse_args()
    
    # Check API key
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found. Set it in .env or environment.", file=sys.stderr)
        sys.exit(1)
    
    # Validate script file
    if not args.script.exists():
        print(f"Error: Script file not found: {args.script}", file=sys.stderr)
        sys.exit(1)
    
    # Read script
    script_text = args.script.read_text(encoding='utf-8')
    
    # Format chapters for the prompt (convert comma-separated to newlines)
    if args.chapters:
        chapters_formatted = "\n".join(c.strip() for c in args.chapters.split(','))
    else:
        chapters_formatted = "(No chapters provided)"
    
    print(f"🤖 Generating description with {GEMINI_MODEL}...", file=sys.stderr)
    
    # Generate description
    description = generate_description_with_gemini(
        script_text=script_text,
        keyword=args.keyword,
        title=args.title,
        chapters=chapters_formatted,
    )
    
    # Output
    if args.output:
        args.output.write_text(description, encoding='utf-8')
        print(f"✅ Description saved to: {args.output}", file=sys.stderr)
    else:
        print(description)
        print("\n" + "=" * 60, file=sys.stderr)
        print(f"📊 Character count: {len(description)} (max 5000)", file=sys.stderr)


if __name__ == "__main__":
    main()
