#!/usr/bin/env python3
"""Captions-only YouTube subscription summarizer (no Whisper fallback).

This script is a lighter variant of `youtube-summ.py` that:
  1. Authenticates with the YouTube Data API (OAuth) to list the user's subscriptions.
  2. Finds videos published *today* (local date) for those channels.
  3. Attempts to fetch official or auto-generated captions using `youtube-transcript-api`.
  4. If captions are available, merges them into plain text and summarizes with Cohere.
  5. Persists / updates entries in an output JSON file (default: `youtube_summaries_captions_only.json`).

Differences vs full pipeline:
  - No audio download.
  - No Whisper / faster-whisper fallback.
  - Videos without captions are skipped silently (with a console note).

Environment variables (optionally via .env):
  COHERE_API_KEY (required)
  COHERE_MODEL (optional, default command-r7b-12-2024)
  YOUTUBE_CLIENT_SECRETS_FILE (optional path to client_secret.json)
  YOUTUBE_SUMMARY_SENTENCES (optional default sentence target)

CLI options: see --help.

Requires: google-api-python-client, google-auth-oauthlib, youtube-transcript-api, requests, python-dotenv.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)
try:
    from youtube_transcript_api.proxies import WebshareProxyConfig  # type: ignore
except Exception:  # pragma: no cover
    WebshareProxyConfig = None  # fallback if proxies module missing
import time
import random
import requests

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_CANDIDATES: Iterable[Path] = (
    ROOT_DIR / ".env",
    ROOT_DIR / "Custom GPTs" / ".env",
)
for candidate in ENV_CANDIDATES:
    if candidate.exists():
        load_dotenv(candidate)
        break
else:
    load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_MODEL = os.getenv("COHERE_MODEL", "command-r7b-12-2024")
DEFAULT_SENTENCES = int(os.getenv("YOUTUBE_SUMMARY_SENTENCES", "6"))

# Optional Webshare rotating residential proxy credentials
WEBSHARE_PROXY_USERNAME = os.getenv("WEBSHARE_PROXY_USERNAME")
WEBSHARE_PROXY_PASSWORD = os.getenv("WEBSHARE_PROXY_PASSWORD")
WEBSHARE_PROXY_COUNTRIES = os.getenv("WEBSHARE_PROXY_COUNTRIES")  # e.g. "us,de,es"

SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]
CLIENT_SECRETS_PATH = Path(os.getenv("YOUTUBE_CLIENT_SECRETS_FILE", ROOT_DIR / "client_secret.json"))
YOUTUBE_CLIENT_ID = os.getenv("YOUTUBE_CLIENT_ID")
YOUTUBE_CLIENT_SECRET = os.getenv("YOUTUBE_CLIENT_SECRET")
TOKEN_PATH = ROOT_DIR / "youtube_token.json"

OUTPUT_JSON = ROOT_DIR / "youtube_summaries_captions_only.json"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def to_local_date(iso_ts: str) -> dt.date:
    parsed = dt.datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    return parsed.astimezone().date()


def load_existing_entries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def persist_entries(path: Path, entries: List[Dict[str, Any]]) -> None:
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(entries, fh, ensure_ascii=False, indent=2)
        print(f"Wrote {len(entries)} summaries to {path}")
    except Exception as exc:
        print(f"Failed to write output: {exc}")


def merge_entries(existing: List[Dict[str, Any]], new_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {e.get("video_id"): e for e in existing if e.get("video_id")}
    for e in new_entries:
        by_id[e.get("video_id")] = e

    def sort_key(item: Dict[str, Any]):
        published_at = item.get("published_at")
        try:
            return dt.datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
        except Exception:
            return dt.datetime.min

    return sorted(by_id.values(), key=sort_key, reverse=True)


# ---------------------------------------------------------------------------
# Auth & YouTube API
# ---------------------------------------------------------------------------

def get_youtube_service():
    creds: Optional[Credentials] = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception as exc:
            print(f"Token refresh failed: {exc}; starting new auth flow.")
            creds = None

    if not creds:
        if CLIENT_SECRETS_PATH.exists():
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS_PATH), SCOPES)
        elif YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET:
            client_config = {
                "installed": {
                    "client_id": YOUTUBE_CLIENT_ID,
                    "client_secret": YOUTUBE_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["http://localhost"],
                }
            }
            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
        else:
            raise FileNotFoundError("Missing YouTube OAuth client configuration.")
        creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())
        print(f"Stored OAuth token at {TOKEN_PATH}")

    return build("youtube", "v3", credentials=creds)


def list_subscribed_channels(youtube, debug: bool) -> List[Dict[str, str]]:
    channels: List[Dict[str, str]] = []
    req = youtube.subscriptions().list(mine=True, part="snippet", maxResults=50, order="alphabetical")
    while req is not None:
        resp = req.execute()
        for item in resp.get("items", []):
            snippet = item.get("snippet", {})
            rid = snippet.get("resourceId", {})
            cid = rid.get("channelId")
            title = snippet.get("title")
            if cid and title:
                channels.append({"id": cid, "title": title})
        req = youtube.subscriptions().list_next(req, resp)
    if debug:
        print(f"[debug] Retrieved {len(channels)} subscribed channels")
    return channels


def get_uploads_playlist_id(youtube, channel_id: str) -> Optional[str]:
    try:
        resp = youtube.channels().list(part="contentDetails", id=channel_id, maxResults=1).execute()
        items = resp.get("items", [])
        if not items:
            return None
        return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    except HttpError as exc:
        print(f"API error getting uploads playlist for {channel_id}: {exc}")
        return None


def list_videos_published_on(youtube, playlist_id: str, target_date: dt.date, debug: bool) -> List[Dict[str, str]]:
    videos: List[Dict[str, str]] = []
    req = youtube.playlistItems().list(playlistId=playlist_id, part="snippet,contentDetails", maxResults=50)
    while req is not None:
        resp = req.execute()
        items = resp.get("items", [])
        for item in items:
            content = item.get("contentDetails", {})
            snippet = item.get("snippet", {})
            vid = content.get("videoId")
            published_at = content.get("videoPublishedAt")
            title = snippet.get("title")
            if not (vid and published_at and title):
                continue
            if to_local_date(published_at) == target_date:
                videos.append({
                    "video_id": vid,
                    "title": title,
                    "published_at": published_at,
                    "description": snippet.get("description", ""),
                    "channel_title": snippet.get("videoOwnerChannelTitle"),
                })
        # Break early if everything in page older than target date
        if items and all(
            to_local_date(i.get("contentDetails", {}).get("videoPublishedAt", "1970-01-01T00:00:00Z")) < target_date
            for i in items
        ):
            break
        req = youtube.playlistItems().list_next(req, resp)
    if debug and videos:
        print(f"[debug] Found {len(videos)} videos on {target_date} in playlist {playlist_id}")
    return videos


# ---------------------------------------------------------------------------
# Caption acquisition (captions only, no Whisper)
# ---------------------------------------------------------------------------

def fetch_caption_transcript(video_id: str, languages: Optional[List[str]] = None, debug: bool = False) -> Optional[str]:
    languages = languages or ["en", "en-US", "es", "es-419"]

    def _entries_to_text(entries):
        lines: List[str] = []
        for e in entries:
            if isinstance(e, dict):
                txt = e.get("text")
            else:
                txt = getattr(e, "text", None)
            if txt:
                lines.append(txt)
        joined = "\n".join(lines).strip()
        return re.sub(r"\s+", " ", joined)

    # Build a fresh proxy config per request to force rotation.
    proxy_cfg = None
    if WEBSHARE_PROXY_USERNAME and WEBSHARE_PROXY_PASSWORD and WebshareProxyConfig:
        chosen_locs = None
        if WEBSHARE_PROXY_COUNTRIES:
            available = [c.strip().lower() for c in WEBSHARE_PROXY_COUNTRIES.split(',') if c.strip()]
            # pick one country per request to encourage proxy rotation
            try:
                chosen = random.choice(available)
                chosen_locs = [chosen]
            except Exception:
                chosen_locs = available or None
        try:
            # Force the known Webshare hostname (proxy.webshare.io) instead of default 'p.webshare.io'
            proxy_cfg = WebshareProxyConfig(
                proxy_username=WEBSHARE_PROXY_USERNAME,
                proxy_password=WEBSHARE_PROXY_PASSWORD,
                filter_ip_locations=chosen_locs if chosen_locs else None,
                domain_name="proxy.webshare.io",
            )
            if debug:
                print(f"[debug] Using Webshare proxy (countries={chosen_locs if chosen_locs else 'ANY'})")
        except Exception as e:
            if debug:
                print(f"[debug] Failed to init Webshare proxy config: {e}")
            proxy_cfg = None

    # Gentle pause to look less like a bot (helps avoid IP bans)
    try:
        time.sleep(2)
    except Exception:
        pass

    try:
        # Prefer instance when proxy active
        if proxy_cfg:
            try:
                api_instance = YouTubeTranscriptApi(proxy_config=proxy_cfg)  # type: ignore
                fetched = api_instance.fetch(video_id)
                if hasattr(fetched, 'to_raw_data'):
                    raw = fetched.to_raw_data()
                    text = _entries_to_text(raw)
                    if text:
                        return text
                text = _entries_to_text(fetched)
                if text:
                    return text
            except Exception as inner:
                if debug:
                    print(f"[debug] proxy instance fetch failed for {video_id}: {inner}")

        # Fast path: class method
        if hasattr(YouTubeTranscriptApi, 'get_transcript'):
            try:
                t = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)  # type: ignore[attr-defined]
                text = _entries_to_text(t)
                if text:
                    return text
            except Exception as inner:
                if debug:
                    print(f"[debug] get_transcript failed for {video_id}: {inner}")

        # Instance path (non-proxy or retry)
        try:
            api_instance = YouTubeTranscriptApi(proxy_config=proxy_cfg) if proxy_cfg else YouTubeTranscriptApi()  # type: ignore
            fetched = api_instance.fetch(video_id)  # type: ignore[attr-defined]
            if hasattr(fetched, 'to_raw_data'):
                raw = fetched.to_raw_data()
                text = _entries_to_text(raw)
                if text:
                    return text
            text = _entries_to_text(fetched)
            if text:
                return text
        except Exception as inner:
            if debug:
                print(f"[debug] instance fetch (retry) failed for {video_id}: {inner}")

        # Enumerate transcripts fallback (NOTE: list_transcripts may ignore proxy internally)
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)  # type: ignore[attr-defined]
            selected = None
            for lang in languages:
                try:
                    selected = transcript_list.find_transcript([lang])
                    if selected:
                        break
                except NoTranscriptFound:
                    continue
            if not selected:
                try:
                    selected = transcript_list.find_generated_transcript(languages)
                except Exception:
                    pass
            if not selected:
                return None
            fetched = selected.fetch()
            return _entries_to_text(fetched) or None
        except Exception as inner:
            if debug:
                print(f"[debug] manual enumeration failed for {video_id}: {inner}")
            return None
    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript, VideoUnavailable):
        return None
    except Exception as exc:
        if debug:
            print(f"Transcript error for {video_id}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Cohere summarization
# ---------------------------------------------------------------------------

def summarize_text(text: str, target_sentences: int, debug: bool) -> Optional[str]:
    if not COHERE_API_KEY:
        raise RuntimeError("COHERE_API_KEY not set.")
    if not text.strip():
        return None

    max_chars = 20000
    trimmed = text if len(text) <= max_chars else text[:max_chars] + "\n\n[truncated]"

    prompt = (
        "You are a YOUTUBE videos summarizer. Summarize the provided video transcription body.\n"
        "Rules:\n"
        f"1. Length: {target_sentences} sentences (rich detail, no filler).\n"
        "2. Use only facts present in the text. No invented people, numbers, links, or events.\n"
        "3. If info is missing/unclear -> output exactly: The text does not contain relevant information.\n"
        "4. Neutral, professional voice.\n"
        "5. Separate by bullet points by topic.\n"
        "Output ONLY the summary text.\n\n"
        f"VIDEO TRANSCRIPTION:\n{trimmed}"
    )

    headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": COHERE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.2,
    }

    if debug:
        print(f"[debug] Cohere request model={COHERE_MODEL} prompt_len={len(prompt)}")
    resp = requests.post("https://api.cohere.com/v2/chat", json=payload, headers=headers, timeout=60)
    if debug:
        print(f"[debug] Cohere HTTP {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()
    summary = data.get("text", "").strip()
    if summary and debug:
        print(f"[debug] Summary excerpt: {summary[:160]}...")
    return summary or None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_today(target_sentences: int, max_videos: Optional[int], debug: bool, show_empty: bool):
    youtube = get_youtube_service()
    channels = list_subscribed_channels(youtube, debug=debug)
    if not channels:
        print("No subscriptions found.")
        return

    today = dt.datetime.now().date()
    pairs: List[Dict[str, Any]] = []
    for ch in channels:
        uploads = get_uploads_playlist_id(youtube, ch["id"])
        if not uploads:
            continue
        vids = list_videos_published_on(youtube, uploads, today, debug=debug)
        if vids:
            for v in vids:
                v["channel_id"] = ch["id"]
                v["channel_title"] = ch["title"]
                pairs.append(v)
        elif debug and show_empty:
            print(f"[debug] No videos today for channel {ch['title']}")

    if not pairs:
        print("No videos uploaded today among your subscriptions.")
        return

    if max_videos is not None:
        pairs = pairs[:max_videos]

    existing = load_existing_entries(OUTPUT_JSON)
    new_entries: List[Dict[str, Any]] = []

    for v in pairs:
        vid = v["video_id"]
        if any(e.get("video_id") == vid for e in existing):
            if debug:
                print(f"[debug] Already summarized {vid}; skipping.")
            continue
        url = f"https://www.youtube.com/watch?v={vid}"
        print(f"Processing: {v['channel_title']} — {v['title']} ({url})")

        caption_text = fetch_caption_transcript(vid, debug=debug)
        if not caption_text:
            print("  No captions available; skipping.")
            continue

        try:
            summary = summarize_text(caption_text, target_sentences=target_sentences, debug=debug)
        except Exception as exc:
            print(f"  Summarization failed: {exc}")
            continue

        new_entries.append({
            "type": "youtube",
            "channel": v.get("channel_title"),
            "channel_id": v.get("channel_id"),
            "video_id": vid,
            "title": v.get("title"),
            "published_at": v.get("published_at"),
            "url": url,
            "transcript_source": "captions",
            "summary": summary,
        })

    if not new_entries:
        print("No new summaries created.")
        return

    combined = merge_entries(existing, new_entries)
    persist_entries(OUTPUT_JSON, combined)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize today's YouTube uploads using only captions (no Whisper).")
    p.add_argument("--sentences", type=int, default=DEFAULT_SENTENCES, help="Target sentences per summary (default: %(default)s)")
    p.add_argument("--max-videos", type=int, default=None, help="Limit number of videos processed.")
    p.add_argument("--debug", action="store_true", help="Enable debug logging.")
    p.add_argument("--show-empty", action="store_true", help="With --debug, also show channels with zero videos today.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    process_today(
        target_sentences=args.sentences,
        max_videos=args.max_videos,
        debug=args.debug,
        show_empty=args.show_empty,
    )


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:
        print("Interrupted by user")
