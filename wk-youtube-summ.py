#!/usr/bin/env python3
"""Summarize today's YouTube subscription uploads using Cohere and faster-whisper.

This script will:
1. Authorize against the YouTube Data API using OAuth (client secrets + token file).
2. Retrieve the channels the authenticated account is subscribed to.
3. Find videos published today for each subscription.
4. Try to obtain official captions via youtube-transcript-api; if unavailable, download audio
   with yt-dlp and generate a transcript using faster-whisper on GPU.
5. Summarize the transcript with Cohere using a bullet-focused prompt.
6. Persist results to ``youtube_summaries.json`` at the workspace root.

Environment variables (load from .env in repo root or Custom GPTs/.env):
- COHERE_API_KEY (required)
- COHERE_MODEL (optional, defaults to command-r7b-12-2024)
- WHISPER_MODEL (optional, defaults to base)
- YOUTUBE_CLIENT_SECRETS_FILE (optional path to OAuth client secrets JSON)
- YOUTUBE_SUMMARY_SENTENCES (optional default sentence count per summary)
- ALLOW_CPU_FALLBACK (set to 1/true to allow slower CPU transcription if CUDA not available)

First run requires a browser-based OAuth flow; credentials are cached in youtube_token.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
from youtube_transcript_api._errors import IpBlocked  # type: ignore

from yt_dlp import YoutubeDL

import torch
from faster_whisper import WhisperModel
import requests
import subprocess
import tempfile
import shutil
import random
import socket
import urllib.parse
import time as _time

###################################################################################################
# Early path setup: define ROOT_DIR before any globals that depend on it (prevents NameError order).
###################################################################################################
ROOT_DIR = Path(__file__).resolve().parents[1]

# (We load .env a few lines later; proxy env vars will be read AFTER .env is loaded for correctness.)


# ---------------------------------------------------------------------------
# Environment & global configuration
# ---------------------------------------------------------------------------

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
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base").lower()
DEFAULT_SENTENCES = int(os.getenv("YOUTUBE_SUMMARY_SENTENCES", "6"))
ALLOW_CPU_FALLBACK = os.getenv("ALLOW_CPU_FALLBACK", "0").lower() in {"1", "true", "yes", "y"}

# ----------------------------------------------------------------------------------
# Webshare / Proxy configuration (used ONLY for transcript scraping; NOT for Google OAuth)
# (Loaded after .env so values from file are available.)
# ----------------------------------------------------------------------------------
WEBSHARE_API_KEY = os.getenv("WEBSHARE_API_KEY") or os.getenv("WEBSHAREAPIKEY")
WEBSHARE_PROXY_USERNAME = os.getenv("WEBSHARE_PROXY_USERNAME") or os.getenv("PROXY_USERNAME")
WEBSHARE_PROXY_PASSWORD = os.getenv("WEBSHARE_PROXY_PASSWORD") or os.getenv("PROXY_PASSWORD")
WEBSHARE_PROXY_COUNTRIES = os.getenv("WEBSHARE_PROXY_COUNTRIES") or os.getenv("PROXY_COUNTRIES")
USE_IP_AUTH = (os.getenv("WEBSHARE_IP_AUTH", "false").lower() in ("1", "true", "yes"))
DISABLE_PROXIES = (os.getenv("DISABLE_WEBSHARE_PROXIES", "false").lower() in ("1", "true", "yes"))

# Cache file for resolved backbone host (shared logic w/ standalone script)
_HOST_CACHE_FILE = ROOT_DIR / "Custom GPTs" / ".webshare_host_cache.json"

# Subscription channel cache (avoid fetching full list every run).
# To force refresh you can:
#   1) Run with the CLI flag:  --refresh-channels
#   2) Manually delete the file below and rerun the script.
SUBS_CACHE_PATH = ROOT_DIR / "youtube_subscriptions_cache.json"
SUBS_CACHE_TTL_DAYS = int(os.getenv("SUBSCRIPTIONS_CACHE_TTL_DAYS", "7"))

# Daily videos cache (today's uploads) to avoid repeated playlist scans if you run multiple
# times in the same day. Delete this file or pass --refresh-today to force a new fetch.
TODAY_VIDEOS_CACHE_PATH = ROOT_DIR / "youtube_today_videos_cache.json"

SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]
CLIENT_SECRETS_PATH = Path(
	os.getenv("YOUTUBE_CLIENT_SECRETS_FILE", ROOT_DIR / "client_secret.json")
)
YOUTUBE_CLIENT_ID = os.getenv("YOUTUBE_CLIENT_ID")
YOUTUBE_CLIENT_SECRET = os.getenv("YOUTUBE_CLIENT_SECRET")
TOKEN_PATH = ROOT_DIR / "youtube_token.json"

DOWNLOAD_DIR = ROOT_DIR / "downloads"
YOUTUBE_AUDIO_DIR = DOWNLOAD_DIR / "youtube_audio"
TRANSCRIPTS_DIR = DOWNLOAD_DIR / "youtube_transcripts"
OUTPUT_JSON = ROOT_DIR / "youtube_summaries.json"

YOUTUBE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

_whisper_model: Optional[WhisperModel] = None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def check_ffmpeg_availability():
	"""Check if ffmpeg is installed and available in the system's PATH."""
	if not shutil.which("ffmpeg"):
		print(
			"Error: ffmpeg is not installed or not found in your system's PATH. "
			"It is required for audio processing.\n"
			"Please install ffmpeg and ensure it's accessible system-wide.\n"
			"You can download it from https://ffmpeg.org/download.html",
			file=sys.stderr,
		)
		sys.exit(1)


def sanitize_filename(name: str, maxlen: int = 180) -> str:
	"""Create a filesystem-safe name with a conservative length cap."""
	sanitized = re.sub(r"[\\/*?:\"<>|]", "", name)
	sanitized = re.sub(r"\s+", " ", sanitized).strip()
	if not sanitized:
		sanitized = "video"
	return sanitized[:maxlen]


def unique_path(path: Path) -> Path:
	"""Return a unique path by appending _1, _2, ... if needed."""
	if not path.exists():
		return path
	stem, suffix = path.stem, path.suffix
	parent = path.parent
	idx = 1
	while True:
		candidate = parent / f"{stem}_{idx}{suffix}"
		if not candidate.exists():
			return candidate
		idx += 1


def to_local_date(iso_ts: str) -> dt.date:
	"""Convert an RFC3339 timestamp to local date."""
	parsed = dt.datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
	return parsed.astimezone().date()


def load_existing_entries(path: Path) -> List[Dict[str, object]]:
	if not path.exists():
		return []
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return []


def persist_entries(path: Path, entries: List[Dict[str, object]]) -> None:
	try:
		with path.open("w", encoding="utf-8") as handle:
			json.dump(entries, handle, ensure_ascii=False, indent=2)
		print(f"Wrote {len(entries)} summaries to {path}")
	except Exception as exc:
		print(f"Failed to write {path}: {exc}")


# ---------------------------------------------------------------------------
# YouTube Data API helpers
# ---------------------------------------------------------------------------

def get_youtube_service() -> Any:
	"""Authorize and return an authenticated YouTube Data API client."""
	creds: Optional[Credentials] = None
	if TOKEN_PATH.exists():
		creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

	if creds and creds.expired and creds.refresh_token:
		try:
			creds.refresh(Request())
		except Exception as exc:
			print(f"Failed to refresh YouTube token: {exc}; starting new auth flow.")
			creds = None

	if not creds:
		if CLIENT_SECRETS_PATH.exists():
			flow = InstalledAppFlow.from_client_secrets_file(
				str(CLIENT_SECRETS_PATH), SCOPES
			)
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
			raise FileNotFoundError(
				"YouTube OAuth client configuration missing. Either set YOUTUBE_CLIENT_SECRETS_FILE "
				"to a client_secret.json path or provide YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET env vars."
			)
		creds = flow.run_local_server(port=0)
		TOKEN_PATH.write_text(creds.to_json())
		print(f"Stored OAuth token at {TOKEN_PATH}")

	return build("youtube", "v3", credentials=creds)


def list_subscribed_channels(youtube, debug: bool = False) -> List[Dict[str, str]]:
	"""Return a list of channel ids and titles for subscriptions."""
	channels: List[Dict[str, str]] = []
	request = youtube.subscriptions().list(
		mine=True, part="snippet", maxResults=50, order="alphabetical"
	)
	while request is not None:
		response = request.execute()
		for item in response.get("items", []):
			snippet = item.get("snippet", {})
			resource = snippet.get("resourceId", {})
			channel_id = resource.get("channelId")
			title = snippet.get("title")
			if channel_id and title:
				channels.append({"id": channel_id, "title": title})
		request = youtube.subscriptions().list_next(request, response)

	if debug:
		print(f"[debug] Retrieved {len(channels)} subscribed channels from live API")
	return channels


# ---------------------------------------------------------------------------
# Channel RSS feed discovery (avoid YouTube Data API for daily video listing)
# ---------------------------------------------------------------------------

def fetch_channel_rss_entries(channel_id: str, timeout: int = 15, debug: bool = False) -> List[Dict[str, str]]:
	"""Fetch channel RSS feed and return list of raw entry dicts.

	Feed URL pattern: https://www.youtube.com/feeds/videos.xml?channel_id=<channel_id>
	Returns simplified dicts with keys: video_id, title, published_at.
	Description is not present in the feed; kept blank for uniform structure.
	"""
	url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
	try:
		resp = requests.get(url, timeout=timeout)
		resp.raise_for_status()
	except Exception as e:
		if debug:
			print(f"[debug] RSS fetch failed for {channel_id}: {e}")
		return []
	import xml.etree.ElementTree as ET  # stdlib
	try:
		root = ET.fromstring(resp.text)
	except Exception as e:
		if debug:
			print(f"[debug] RSS parse failed for {channel_id}: {e}")
		return []
	# Namespace handling
	ns = {"yt": "http://www.youtube.com/xml/schemas/2015", "media": "http://search.yahoo.com/mrss/"}
	entries: List[Dict[str, str]] = []
	for entry in root.findall("entry"):
		vid_el = entry.find("yt:videoId", ns)
		title_el = entry.find("title")
		pub_el = entry.find("published")
		if vid_el is None or title_el is None or pub_el is None:
			continue
		video_id = vid_el.text or ""
		title = title_el.text or ""
		published_at = pub_el.text or ""
		if not video_id or not title or not published_at:
			continue
		entries.append({
			"video_id": video_id,
			"title": title,
			"published_at": published_at,
			"description": "",
			"channel_title": "",
		})
	if debug:
		print(f"[debug] RSS entries for {channel_id}: {len(entries)}")
	return entries


def collect_today_videos_rss(channels: List[Dict[str, str]], target_date: dt.date, debug: bool = False) -> List[Tuple[Dict[str, str], Dict[str, str]]]:
	"""Collect today's videos using channel RSS feeds instead of the YouTube Data API.

	This avoids consuming API quota and can be used when channel list is already cached.
	Note: RSS publishes in UTC; date comparison uses local date conversion via to_local_date.
	"""
	pairs: List[Tuple[Dict[str, str], Dict[str, str]]] = []
	for channel in channels:
		cid = channel.get("id")
		if not cid:
			continue
		raw_entries = fetch_channel_rss_entries(cid, debug=debug)
		for entry in raw_entries:
			try:
				entry_date = to_local_date(entry["published_at"])
			except Exception:
				continue
			if entry_date == target_date:
				pairs.append((channel, entry))
	if debug:
		print(f"[debug] RSS discovered {len(pairs)} videos for today")
	return pairs


def load_cached_subscriptions(refresh: bool = False, debug: bool = False) -> Optional[List[Dict[str, str]]]:
	if refresh or not SUBS_CACHE_PATH.exists():
		return None
	try:
		raw = json.loads(SUBS_CACHE_PATH.read_text(encoding="utf-8"))
		fetched_at = raw.get("fetched_at")
		channels = raw.get("channels")
		if not isinstance(channels, list):
			return None
		if fetched_at:
			try:
				ts = dt.datetime.fromisoformat(fetched_at)
				if dt.datetime.now() - ts > dt.timedelta(days=SUBS_CACHE_TTL_DAYS):
					if debug:
						print("[debug] Subscription cache expired by TTL")
					return None
			except Exception:
				return None
		if debug:
			print(f"[debug] Loaded {len(channels)} subscriptions from cache {SUBS_CACHE_PATH}")
		return channels
	except Exception as e:
		if debug:
			print(f"[debug] Failed loading subscription cache: {e}")
		return None


def persist_subscription_cache(channels: List[Dict[str, str]], debug: bool = False) -> None:
	try:
		SUBS_CACHE_PATH.write_text(
			json.dumps({"fetched_at": dt.datetime.now().isoformat(), "channels": channels}, ensure_ascii=False, indent=2),
			encoding="utf-8",
		)
		if debug:
			print(f"[debug] Wrote subscription cache to {SUBS_CACHE_PATH}")
	except Exception as e:
		if debug:
			print(f"[debug] Failed writing subscription cache: {e}")


def load_today_videos_cache(target_date: dt.date, refresh: bool = False, debug: bool = False) -> Optional[List[Dict[str, str]]]:
	"""Load cached list of today's (date-labeled) channel-video pairs.

	File schema:
	{
	  "date": "YYYY-MM-DD",
	  "videos": [ {channel_id, channel_title, video_id, title, published_at, description}, ... ]
	}
	"""
	if refresh or not TODAY_VIDEOS_CACHE_PATH.exists():
		return None
	try:
		raw = json.loads(TODAY_VIDEOS_CACHE_PATH.read_text(encoding="utf-8"))
		if raw.get("date") != target_date.isoformat():
			return None
		vids = raw.get("videos")
		if isinstance(vids, list):
			if debug:
				print(f"[debug] Loaded {len(vids)} videos from daily cache {TODAY_VIDEOS_CACHE_PATH}")
			return vids
	except Exception as e:
		if debug:
			print(f"[debug] Failed reading today videos cache: {e}")
	return None


def persist_today_videos_cache(target_date: dt.date, pairs: List[Tuple[Dict[str, str], Dict[str, str]]], debug: bool = False) -> None:
	try:
		flat: List[Dict[str, str]] = []
		for channel, video in pairs:
			flat.append({
				"channel_id": channel.get("id", ""),
				"channel_title": channel.get("title", ""),
				"video_id": video.get("video_id", ""),
				"title": video.get("title", ""),
				"published_at": video.get("published_at", ""),
				"description": video.get("description", "")
			})
		TODAY_VIDEOS_CACHE_PATH.write_text(
			json.dumps({"date": target_date.isoformat(), "videos": flat}, ensure_ascii=False, indent=2),
			encoding="utf-8"
		)
		if debug:
			print(f"[debug] Wrote today videos cache with {len(flat)} entries to {TODAY_VIDEOS_CACHE_PATH}")
	except Exception as e:
		if debug:
			print(f"[debug] Failed writing today videos cache: {e}")


def get_uploads_playlist_id(youtube, channel_id: str) -> Optional[str]:
	try:
		resp = youtube.channels().list(
			part="contentDetails", id=channel_id, maxResults=1
		).execute()
		items = resp.get("items", [])
		if not items:
			return None
		return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
	except HttpError as exc:
		print(f"YouTube API error fetching uploads playlist for {channel_id}: {exc}")
		return None


def list_videos_published_on(
    youtube,
    playlist_id: str,
    target_date: dt.date,
    debug: bool = False,
    show_empty: bool = False,
) -> List[Dict[str, str]]:
    """Return playlist items published on the given date.

    Notes:
        - Every channel has a special implicit *uploads* playlist (channel ID UCxxxx -> playlist ID UUxxxx).
        - We stop paging once all items in a page are older than the target date to reduce API calls.
    """
    videos: List[Dict[str, str]] = []
    request = youtube.playlistItems().list(
        playlistId=playlist_id,
        part="snippet,contentDetails",
        maxResults=50,
    )

    while request is not None:
        response = request.execute()
        items = response.get("items", [])
        for item in items:
            content = item.get("contentDetails", {})
            video_id = content.get("videoId")
            published_at = content.get("videoPublishedAt")
            snippet = item.get("snippet", {})
            title = snippet.get("title")
            if not (video_id and published_at and title):
                continue
            video_date = to_local_date(published_at)
            if video_date == target_date:
                videos.append(
                    {
                        "video_id": video_id,
                        "title": title,
                        "published_at": published_at,
                        "description": snippet.get("description", ""),
                        "channel_title": snippet.get("videoOwnerChannelTitle"),
                    }
                )
        if items and all(
            to_local_date(i["contentDetails"]["videoPublishedAt"]) < target_date
            for i in items
            if "contentDetails" in i and "videoPublishedAt" in i["contentDetails"]
        ):
            break
        request = youtube.playlistItems().list_next(request, response)

    if debug and (show_empty or videos):
        print(f"[debug] Found {len(videos)} videos on {target_date} in playlist {playlist_id}")
    return videos


# ---------------------------------------------------------------------------
# Transcript acquisition
# ---------------------------------------------------------------------------

def fetch_caption_transcript(video_id: str, languages: Optional[List[str]] = None, debug: bool = False) -> Optional[str]:
	"""Attempt to fetch an official caption transcript.

	Handles older/newer versions of youtube-transcript-api where attribute layouts may differ.
	Falls back to iterating transcript list if get_transcript is absent.
	"""
	languages = languages or ["en", "en-US", "es", "es-419"]

	def _entries_to_text(entries: Iterable[Any]) -> str:
		lines: List[str] = []
		for entry in entries:
			# New API: objects with .text; Old API: dicts with 'text'
			if isinstance(entry, dict):
				val = entry.get("text")
			else:
				val = getattr(entry, "text", None)
			if val:
				lines.append(val)
		return "\n".join(lines).strip()

	# Strategy:
	# 1. Try classmethod get_transcript (old API) -> may return list[dict] or new object iterable
	# 2. Try instance .fetch() (new 1.2.x API) returning FetchedTranscript
	# 3. Fallback enumerate transcripts & fetch chosen one
	try:
		# Quick guard: if proxies explicitly enabled-first strategy not required; we still try direct first
		if hasattr(YouTubeTranscriptApi, "get_transcript"):
			try:
				transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)  # type: ignore[attr-defined]
				text = _entries_to_text(transcript)
				if text:
					return text
			except Exception as inner:
				if debug:
					print(f"[debug] get_transcript path failed: {inner}")

		# Newer API object style
		try:
			api_instance = YouTubeTranscriptApi()  # type: ignore[call-arg]
			fetched_obj = api_instance.fetch(video_id)  # type: ignore[attr-defined]
			if hasattr(fetched_obj, "to_raw_data"):
				raw = fetched_obj.to_raw_data()
				text = _entries_to_text(raw)
				if text:
					return text
			# Fallback iterate directly
			text = _entries_to_text(fetched_obj)
			if text:
				return text
		except Exception as inner:
			if debug:
				print(f"[debug] instance fetch path failed: {inner}")

		# Enumerate transcripts manually
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
			text = _entries_to_text(fetched)
			return text or None
		except Exception as inner:
			if debug:
				print(f"[debug] manual enumeration path failed: {inner}")
			return None
	except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript, VideoUnavailable):
		return None
	except Exception as exc:
		if debug:
			print(f"Transcript retrieval error for {video_id}: {exc}")
		else:
			print(f"Transcript retrieval error for {video_id}: {exc.__class__.__name__}: {exc}")
		return None


# ---------------- Proxy-enabled transcript fetch (subset of wk-pruebyttranscriptapi logic) ---------------- #

def _doh_query(host: str, provider: str, timeout: float = 4.0) -> List[str]:
	try:
		if "dns.google" in provider:
			r = requests.get(provider, params={"name": host, "type": "A"}, timeout=timeout)
		else:
			r = requests.get(provider, params={"name": host, "type": "A"}, headers={"accept": "application/dns-json"}, timeout=timeout)
		j = r.json()
		answers = j.get("Answer", [])
		ips = [a.get("data") for a in answers if a.get("type") == 1]
		return [ip for ip in ips if ip and ip != "0.0.0.0"]
	except Exception:
		return []


def resolve_webshare_host(host: str = "p.webshare.io", ttl: int = 3600) -> str:
	try:
		infos = socket.getaddrinfo(host, 80, proto=socket.IPPROTO_TCP)
		ips = sorted({i[4][0] for i in infos if i[4][0] and i[4][0] != "0.0.0.0"})
		if ips:
			return host
	except socket.gaierror:
		pass
	now = int(_time.time())
	if _HOST_CACHE_FILE.exists():
		try:
			cache = json.loads(_HOST_CACHE_FILE.read_text(encoding="utf-8"))
			if cache.get("host") == host and isinstance(cache.get("ts"), int) and now - cache["ts"] <= ttl:
				cached_ip = cache.get("ip")
				if cached_ip:
					return cached_ip
		except Exception:
			pass
	providers = ["https://dns.google/resolve", "https://cloudflare-dns.com/dns-query"]
	for prov in providers:
		ips = _doh_query(host, prov)
		if ips:
			chosen = ips[0]
			try:
				_HOST_CACHE_FILE.write_text(json.dumps({"host": host, "ip": chosen, "ts": now}), encoding="utf-8")
			except Exception:
				pass
			return chosen
	return host


def _substitute_host_in_proxy_url(proxy_url: str, new_host: str) -> str:
	try:
		parsed = urllib.parse.urlparse(proxy_url)
		if not parsed.hostname or parsed.hostname == new_host:
			return proxy_url
		if parsed.hostname not in ("p.webshare.io", "proxy.webshare.io"):
			return proxy_url
		userinfo = ""
		if parsed.username:
			userinfo = parsed.username
			if parsed.password:
				userinfo += f":{parsed.password}"
			userinfo += "@"
		port_part = f":{parsed.port}" if parsed.port else ""
		new_netloc = f"{userinfo}{new_host}{port_part}"
		return urllib.parse.urlunparse(parsed._replace(netloc=new_netloc))
	except Exception:
		return proxy_url


def get_backbone_proxies_from_api(page_size: int = 25, max_pages: int = 2, timeout: int = 10) -> List[str]:
	if not WEBSHARE_API_KEY:
		return []
	base = "https://proxy.webshare.io/api/v2/proxy/list/"
	proxies: List[str] = []
	for page in range(1, max_pages + 1):
		params = {"mode": "backbone", "page": page, "page_size": page_size}
		if WEBSHARE_PROXY_COUNTRIES:
			countries = ",".join([c.strip().upper() for c in WEBSHARE_PROXY_COUNTRIES.split(",") if c.strip()])
			params["country_code__in"] = countries
		try:
			r = requests.get(base, headers={"Authorization": f"Token {WEBSHARE_API_KEY}"}, params=params, timeout=timeout)
			r.raise_for_status()
			j = r.json()
			results = j.get("results") if isinstance(j, dict) else j
			if not results:
				continue
			for entry in results:
				port = entry.get("port") or entry.get("proxy_port")
				if not port:
					continue
				if USE_IP_AUTH:
					proxies.append(f"http://p.webshare.io:{port}")
				else:
					username = entry.get("username") or entry.get("proxy_username") or WEBSHARE_PROXY_USERNAME
					password = entry.get("password") or entry.get("proxy_password") or WEBSHARE_PROXY_PASSWORD
					if not (username and password):
						continue
					host = entry.get("proxy_address") or entry.get("proxy_host") or entry.get("host") or entry.get("ip") or "p.webshare.io"
					if WEBSHARE_PROXY_COUNTRIES and host in ("p.webshare.io", "proxy.webshare.io"):
						try:
							choices = [c.strip() for c in WEBSHARE_PROXY_COUNTRIES.split(",") if c.strip()]
							if choices:
								username = f"{username}-{random.choice(choices).lower()}"
						except Exception:
							pass
					proxies.append(f"http://{username}:{password}@{host}:{port}")
		except Exception:
			continue
	return proxies


def fetch_transcript_via_proxies(video_id: str, languages_priority: List[tuple[str, bool]] | None = None, max_attempts: int = 10, debug: bool = False) -> Optional[str]:
	if DISABLE_PROXIES:
		return None
	languages_priority = languages_priority or [("en", False), ("es", False), ("en", True), ("es", True)]
	proxy_pool = get_backbone_proxies_from_api(page_size=25, max_pages=3)
	if not proxy_pool:
		if debug:
			print("[debug] No proxies from API")
		return None
	resolved = resolve_webshare_host("p.webshare.io")
	attempt = 0
	last_error = None
	while attempt < max_attempts:
		attempt += 1
		proxy_url = random.choice(proxy_pool)
		if resolved and resolved != "p.webshare.io":
			proxy_url = _substitute_host_in_proxy_url(proxy_url, resolved)
		if debug:
			print(f"[debug] proxy attempt {attempt}/{max_attempts} using {proxy_url}")
		env_old_http = os.environ.get("HTTP_PROXY")
		env_old_https = os.environ.get("HTTPS_PROXY")
		try:
			os.environ["HTTP_PROXY"] = proxy_url
			os.environ["HTTPS_PROXY"] = proxy_url
			api = YouTubeTranscriptApi()
			transcript_list = api.list(video_id)
			selected = None
			for lang, gen in languages_priority:
				for t in transcript_list:
					if t.language_code == lang and t.is_generated == gen:
						selected = t
						break
				if selected:
					break
			if not selected and transcript_list:
				selected = transcript_list[0]
			if not selected:
				continue
			data = selected.fetch()
			# Merge text
			parts = []
			for item in data:
				if isinstance(item, dict):
					txt = item.get("text")
				else:
					txt = getattr(item, "text", None)
				if txt:
					parts.append(txt.strip())
			merged = "\n".join(parts).strip()
			if merged:
				return merged
		except IpBlocked as e:
			last_error = e
			if debug:
				print("[debug] IpBlocked – rotating proxy")
			continue
		except Exception as e:
			last_error = e
			if debug:
				print(f"[debug] proxy transcript failure: {type(e).__name__}: {str(e)[:140]}")
			continue
		finally:
			if "HTTP_PROXY" in os.environ:
				if env_old_http is None:
					os.environ.pop("HTTP_PROXY", None)
				else:
					os.environ["HTTP_PROXY"] = env_old_http
			if "HTTPS_PROXY" in os.environ:
				if env_old_https is None:
					os.environ.pop("HTTPS_PROXY", None)
				else:
					os.environ["HTTPS_PROXY"] = env_old_https
	if debug and last_error:
		print(f"[debug] Exhausted proxy attempts; last error: {last_error}")
	return None


def download_youtube_audio(video_id: str, title: str) -> Optional[Path]:
	url = f"https://www.youtube.com/watch?v={video_id}"
	safe_title = sanitize_filename(f"{title}-{video_id}")
	output_template = str((YOUTUBE_AUDIO_DIR / safe_title).with_suffix(".%(ext)s"))
	ydl_opts = {
		"format": "bestaudio/best",
		"outtmpl": output_template,
		"quiet": True,
		"noplaylist": True,
		"postprocessors": [
			{
				"key": "FFmpegExtractAudio",
				"preferredcodec": "mp3",
				"preferredquality": "192",
			},
			{
				"key": "FFmpegPostProcessor",
				"when": "post_process",
				"extra_args": ["-ar", "16000", "-ac", "1"],
			}
		],
	}
	try:
		with YoutubeDL(ydl_opts) as ydl:
			info = ydl.extract_info(url, download=True)
			downloaded = Path(ydl.prepare_filename(info)).with_suffix(".mp3")
			if downloaded.exists():
				return downloaded
			# fall back to scanning directory for matching file
			candidates = list(YOUTUBE_AUDIO_DIR.glob(f"{safe_title}*.mp3"))
			return candidates[-1] if candidates else None
	except Exception as exc:
		print(f"Failed to download audio for {video_id}: {exc}")
		return None


def _debug_cuda_report():
	try:
		print(f"[debug] torch version: {torch.__version__}")
		print(f"[debug] torch.version.cuda: {torch.version.cuda}")
		print(f"[debug] CUDA available: {torch.cuda.is_available()}")
		if torch.cuda.is_available():
			print(f"[debug] device count: {torch.cuda.device_count()}")
			try:
				print(f"[debug] device 0 name: {torch.cuda.get_device_name(0)}")
			except Exception:
				pass
	except Exception as e:
		print(f"[debug] CUDA diagnostic failed: {e}")


def split_audio_into_chunks(audio_path: Path, chunk_seconds: int = 600) -> List[Path]:
	"""Split audio into chunks using ffmpeg. Returns list of chunk file paths."""
	chunks = []
	chunk_dir = audio_path.parent / f"{audio_path.stem}_chunks"
	chunk_dir.mkdir(exist_ok=True)
	
	try:
		# First, ensure the audio is mono 16kHz WAV for consistent processing
		temp_wav = chunk_dir / f"{audio_path.stem}_temp.wav"
		subprocess.run([
			"ffmpeg", "-i", str(audio_path), "-ar", "16000", "-ac", "1", 
			"-y", str(temp_wav)
		], check=True, capture_output=True)
		
		# Split into chunks
		chunk_pattern = str(chunk_dir / f"chunk_%03d.wav")
		subprocess.run([
			"ffmpeg", "-i", str(temp_wav), "-f", "segment", 
			"-segment_time", str(chunk_seconds),
			"-c:a", "pcm_s16le", "-y", chunk_pattern
		], check=True, capture_output=True)
		
		# Clean up temp file
		temp_wav.unlink(missing_ok=True)
		
		# Find all generated chunks
		chunks = sorted(chunk_dir.glob("chunk_*.wav"))
		return chunks
		
	except subprocess.CalledProcessError as e:
		print(f"FFmpeg error splitting audio: {e.stderr.decode() if e.stderr else str(e)}")
		return []
	except Exception as e:
		print(f"Error splitting audio into chunks: {e}")
		return []


def get_whisper_model(model_name: Optional[str] = None, debug: bool = False) -> WhisperModel:
	global _whisper_model
	# Use provided model name or fall back to global default
	model_to_use = model_name or WHISPER_MODEL_NAME
	
	# Reset cached model if we're requesting a different one
	if _whisper_model is not None and hasattr(_whisper_model, '_model_name'):
		if getattr(_whisper_model, '_model_name', None) != model_to_use:
			_whisper_model = None
	
	if _whisper_model is not None:
		return _whisper_model

	have_cuda = torch.cuda.is_available()
	if not have_cuda and debug:
		_debug_cuda_report()

	if not have_cuda:
		if ALLOW_CPU_FALLBACK:
			if debug:
				print("[debug] CUDA not available; using CPU fallback (this will be slow). Set ALLOW_CPU_FALLBACK=0 to disable.")
			device = "cpu"
			# Use an int8 compute type for speed/memory on CPU if supported
			compute_type = "int8" if model_to_use not in {"large", "large-v2", "large-v3"} else "int8_float16"
		else:
			raise RuntimeError(
				"CUDA is not available. GPU is required for faster-whisper transcription. "
				"(Set ALLOW_CPU_FALLBACK=1 to allow slow CPU transcription.)"
			)
	else:
		device = "cuda"
		compute_type = "float16"

	_whisper_model = WhisperModel(
		model_to_use,
		device=device,
		compute_type=compute_type,
	)
	# Store the model name for cache invalidation
	_whisper_model._model_name = model_to_use
	print(f"Loaded faster-whisper '{model_to_use}' ({compute_type}) on {device}.")
	return _whisper_model


def transcribe_with_whisper(audio_path: Path, model_name: Optional[str] = None, chunk_seconds: int = 600) -> Optional[str]:
	def try_transcribe_single(path: Path, model_name_to_use: str) -> Optional[str]:
		"""Attempt to transcribe a single file with the given model."""
		try:
			model = get_whisper_model(model_name_to_use)
			segments, info = model.transcribe(str(path), beam_size=5)
			return "".join(getattr(seg, "text", "") for seg in segments)
		except Exception as exc:
			if "allocate" in str(exc).lower() or "memory" in str(exc).lower():
				print(f"Memory error with model '{model_name_to_use}': {exc}")
				return None  # Signal OOM
			else:
				print(f"Transcription error with model '{model_name_to_use}': {exc}")
				raise exc
	
	def try_transcribe_chunked(path: Path, model_name_to_use: str) -> Optional[str]:
		"""Transcribe by splitting into chunks."""
		print(f"Attempting chunked transcription with {chunk_seconds}s chunks...")
		chunks = split_audio_into_chunks(path, chunk_seconds)
		if not chunks:
			return None
		
		try:
			model = get_whisper_model(model_name_to_use)
			all_text = []
			for i, chunk_path in enumerate(chunks):
				try:
					segments, info = model.transcribe(str(chunk_path), beam_size=5)
					chunk_text = "".join(getattr(seg, "text", "") for seg in segments)
					all_text.append(chunk_text)
					print(f"Transcribed chunk {i+1}/{len(chunks)}")
				except Exception as exc:
					print(f"Failed to transcribe chunk {i+1}: {exc}")
					continue  # Skip this chunk, continue with others
			
			# Clean up chunk directory
			chunk_dir = chunks[0].parent
			shutil.rmtree(chunk_dir, ignore_errors=True)
			
			return " ".join(all_text) if all_text else None
		except Exception as exc:
			print(f"Chunked transcription failed: {exc}")
			return None
	
	# Model fallback order: requested -> base -> small -> tiny
	model_fallbacks = []
	initial_model = model_name or WHISPER_MODEL_NAME
	for m in [initial_model, "base", "small", "tiny"]:
		if m not in model_fallbacks:
			model_fallbacks.append(m)
	
	# Try transcription strategies in order
	for model_to_try in model_fallbacks:
		# Strategy 1: Single file transcription
		try:
			result = try_transcribe_single(audio_path, model_to_try)
			if result is not None:
				# Save transcript
				txt_path = audio_path.with_suffix(".txt")
				txt_path.write_text(result, encoding="utf-8")
				return result
		except Exception:
			pass  # Try next strategy
		
		# Strategy 2: Chunked transcription with same model
		try:
			result = try_transcribe_chunked(audio_path, model_to_try)
			if result is not None:
				# Save transcript
				txt_path = audio_path.with_suffix(".txt")
				txt_path.write_text(result, encoding="utf-8")
				return result
		except Exception:
			pass  # Try next model
		
		print(f"Model '{model_to_try}' failed, trying next fallback...")
	
	print(f"All transcription strategies failed for {audio_path}")
	return None


def get_or_create_transcript(video: Dict[str, str], whisper_model: Optional[str] = None, chunk_seconds: int = 600, debug: bool = False) -> Tuple[Optional[str], Optional[Path], Optional[str]]:
	"""Return transcript text, path, and source label (captions|whisper|cached)."""
	video_id = video["video_id"]
	title = video["title"]
	transcript_path = TRANSCRIPTS_DIR / f"{sanitize_filename(title)}-{video_id}.txt"
	if transcript_path.exists():
		if debug:
			print(f"[debug] Using cached transcript {transcript_path}")
		return transcript_path.read_text(encoding="utf-8"), transcript_path, "cached"

	# 1. Try direct (no proxy) captions
	text = fetch_caption_transcript(video_id, debug=debug)
	if not text:
		# 2. Try proxy-backed retrieval (only for captions; does not affect YouTube Data API)
		if debug:
			print("[debug] Direct caption fetch failed, attempting proxy-backed transcript retrieval...")
		try:
			text = fetch_transcript_via_proxies(video_id, debug=debug)
			if text and debug:
				print("[debug] Proxy-backed caption retrieval succeeded")
		except Exception as e:
			if debug:
				print(f"[debug] Proxy-backed attempt error: {e}")
	if text:
		transcript_path.write_text(text, encoding="utf-8")
		if debug:
			print(f"[debug] Saved caption transcript to {transcript_path}")
		return text, transcript_path, "captions"

	audio_path = download_youtube_audio(video_id, title)
	if not audio_path:
		return None, None, None
	text = transcribe_with_whisper(audio_path, model_name=whisper_model, chunk_seconds=chunk_seconds)
	if text:
		transcript_path.write_text(text, encoding="utf-8")
		if debug:
			print(f"[debug] Whisper transcript saved to {transcript_path}")
		return text, transcript_path, "whisper"
	return None, None, None


# ---------------------------------------------------------------------------
# Cohere summarization
# ---------------------------------------------------------------------------

def summarize_video_transcript(
	text: str,
	target_sentences: int = DEFAULT_SENTENCES,
	debug: bool = False,
) -> Optional[str]:
	if not COHERE_API_KEY:
		raise RuntimeError("COHERE_API_KEY is not configured. Set it in your .env file.")
	if not text or not text.strip():
		return "The text does not contain relevant information."

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
		f"video TRANSCRIPTION:\n{trimmed}"
	)

	model_to_use = COHERE_MODEL or "command-r7b-12-2024"
	headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
	payload = {
		"model": model_to_use,
		"messages": [{"role": "user", "content": prompt}],
		"max_tokens": 512,
		"temperature": 0.2,
	}

	if debug:
		masked = COHERE_API_KEY[:4] + "..." + COHERE_API_KEY[-4:]
		print(f"[debug] Cohere request model={model_to_use}, key={masked}, prompt_len={len(prompt)}")

	resp = requests.post("https://api.cohere.com/v2/chat", json=payload, headers=headers, timeout=60)
	if debug:
		print(f"[debug] Cohere HTTP {resp.status_code}")
	resp.raise_for_status()
	data = resp.json()

	# Cohere responses can differ: legacy 'text', or 'message':{'content':[{'text':...}]}, or 'output_text'
	candidates: List[str] = []
	if isinstance(data, dict):
		for key in ("text", "output_text"):
			val = data.get(key)
			if isinstance(val, str) and val.strip():
				candidates.append(val.strip())
		message = data.get("message")
		if isinstance(message, dict):
			content = message.get("content")
			if isinstance(content, list):
				for part in content:
					if isinstance(part, dict):
						pt = part.get("text")
						if isinstance(pt, str) and pt.strip():
							candidates.append(pt.strip())
	# Deduplicate, keep longest (often most complete)
	unique = sorted({c for c in candidates if c}, key=len, reverse=True)
	summary = unique[0] if unique else ""

	if not summary and debug:
		print(f"[debug] Could not parse summary; raw snippet: {json.dumps(data)[:600]}")

	if summary:
		if debug:
			print(f"[debug] Cohere summary excerpt: {summary[:200]}...")
		return summary
	return None


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def collect_today_videos(
    youtube,
    channels: List[Dict[str, str]],
    target_date: dt.date,
    debug: bool = False,
    show_empty: bool = False,
) -> List[Tuple[Dict[str, str], Dict[str, str]]]:
    pairs: List[Tuple[Dict[str, str], Dict[str, str]]] = []
    for channel in channels:
        playlist_id = get_uploads_playlist_id(youtube, channel["id"])
        if not playlist_id:
            continue
        videos = list_videos_published_on(
            youtube, playlist_id, target_date, debug=debug, show_empty=show_empty
        )
        for video in videos:
            pairs.append((channel, video))
    return pairs
    


def merge_entries(
	existing: List[Dict[str, object]],
	new_entries: List[Dict[str, object]],
) -> List[Dict[str, object]]:
	by_video = {entry.get("video_id"): entry for entry in existing if entry.get("video_id")}
	for entry in new_entries:
		by_video[entry.get("video_id")] = entry
	# sort newest first by published time when present
	def sort_key(item: Dict[str, object]):
		published_at = item.get("published_at")
		try:
			return dt.datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
		except Exception:
			return dt.datetime.min
	return sorted(by_video.values(), key=sort_key, reverse=True)


def summarize_today_uploads(
    target_sentences: int,
    max_videos: Optional[int],
    debug: bool,
    show_empty: bool,
    whisper_model: Optional[str] = None,
    chunk_seconds: int = 600,
	refresh_channels: bool = False,
	refresh_today: bool = False,
	rss_only: bool = False,
	fill_missing: bool = False,
) -> None:
	youtube = get_youtube_service()
	# Attempt cached subscriptions first
	channels = load_cached_subscriptions(refresh=refresh_channels, debug=debug)
	if channels is None:
		channels = list_subscribed_channels(youtube, debug=debug)
		# Persist cache (only if we actually got some)
		if channels:
			persist_subscription_cache(channels, debug=debug)
	if not channels:
		print("No subscriptions found for the authenticated account.")
		return

	today = dt.datetime.now().date()

	# Attempt daily video cache (channel/video pairs flattened) to avoid re-hitting playlists.
	cached_flat = load_today_videos_cache(today, refresh=refresh_today, debug=debug)
	if cached_flat is not None:
		# Reconstruct pairs structure minimally (channel dict + video dict)
		channel_video_pairs: List[Tuple[Dict[str, str], Dict[str, str]]] = []
		for v in cached_flat:
			channel_video_pairs.append(({"id": v.get("channel_id", ""), "title": v.get("channel_title", "")}, {
				"video_id": v.get("video_id", ""),
				"title": v.get("title", ""),
				"published_at": v.get("published_at", ""),
				"description": v.get("description", ""),
				"channel_title": v.get("channel_title", ""),
			}))
		if debug:
			print(f"[debug] Using cached today videos list with {len(channel_video_pairs)} entries")
	else:
		if rss_only:
			channel_video_pairs = collect_today_videos_rss(channels, today, debug=debug)
		else:
			channel_video_pairs = collect_today_videos(
				youtube, channels, today, debug=debug, show_empty=show_empty
			)
		# Persist if we fetched fresh
		persist_today_videos_cache(today, channel_video_pairs, debug=debug)
	if not channel_video_pairs:
		print("No videos uploaded today among your subscriptions.")
		return

	if max_videos is not None:
		channel_video_pairs = channel_video_pairs[:max_videos]

	existing_entries = load_existing_entries(OUTPUT_JSON)
	new_entries: List[Dict[str, object]] = []

	# If fill_missing is requested, filter channel_video_pairs down to ones missing summary already
	if fill_missing and existing_entries:
		missing_ids = {e.get("video_id") for e in existing_entries if not e.get("summary")}
		if missing_ids:
			channel_video_pairs = [pair for pair in channel_video_pairs if pair[1].get("video_id") in missing_ids]
			if debug:
				print(f"[debug] fill-missing mode: retrying {len(channel_video_pairs)} videos with null summaries")

	for channel, video in channel_video_pairs:
		url = f"https://www.youtube.com/watch?v={video['video_id']}"
		print(f"Processing: {channel['title']} — {video['title']} ({url})")

		# Only skip if an existing NON-EMPTY summary is present. If summary is null/empty, reprocess.
		existing_entry = next((e for e in existing_entries if e.get("video_id") == video["video_id"]), None)
		if existing_entry:
			existing_summary = existing_entry.get("summary")
			if isinstance(existing_summary, str) and existing_summary.strip():
				print("  Summary already exists; skipping.")
				continue
			else:
				if debug:
					print("[debug] Existing entry has null/empty summary -> re-summarizing")

		text, transcript_path, source = get_or_create_transcript(video, whisper_model=whisper_model, chunk_seconds=chunk_seconds, debug=debug)
		if not text:
			print("  Unable to obtain transcript; skipping.")
			continue

		try:
			summary = summarize_video_transcript(text, target_sentences=target_sentences, debug=debug)
		except Exception as exc:
			print(f"  Summarization failed: {exc}")
			continue

		if not summary:
			print("  Empty summary returned; skipping entry (will retry next run).")
			continue

		new_entries.append(
			{
				"type": "youtube",
				"channel": channel["title"],
				"channel_id": channel["id"],
				"video_id": video["video_id"],
				"title": video["title"],
				"published_at": video["published_at"],
				"url": url,
				"transcript_file": str(transcript_path) if transcript_path else None,
				"transcript_source": source,
				"summary": summary,
			}
		)

	if not new_entries:
		print("No new summaries created today.")
		return

	combined = merge_entries(existing_entries, new_entries)
	persist_entries(OUTPUT_JSON, combined)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Summarize today's YouTube subscription uploads.")
	parser.add_argument(
		"--sentences",
		type=int,
		default=DEFAULT_SENTENCES,
		help="Target number of sentences per summary (default: %(default)s)",
	)
	parser.add_argument(
		"--max-videos",
		type=int,
		default=None,
		help="Optional cap on number of videos to process (useful for testing).",
	)
	parser.add_argument(
		"--debug",
		action="store_true",
		help="Enable verbose debug logging for API calls and parsing.",
	)
	parser.add_argument(
		"--show-empty",
		action="store_true",
		help="When used with --debug, also print playlists that had zero videos today (default hides them)",
	)
	parser.add_argument(
		"--whisper-model",
		type=str,
		default=None,
		help="Whisper model to use (tiny, small, base, medium, large). Overrides WHISPER_MODEL env var.",
	)
	parser.add_argument(
		"--chunk-seconds",
		type=int,
		default=600,
		help="Split long audio into chunks of this many seconds to avoid OOM (default: %(default)s)",
	)
	parser.add_argument(
		"--refresh-channels",
		action="store_true",
		help="Ignore channel subscription cache and fetch fresh list (also updates cache).",
	)
	parser.add_argument(
		"--refresh-today",
		action="store_true",
		help="Ignore today's uploads cache and re-fetch playlists (updates daily cache).",
	)
	parser.add_argument(
		"--no-proxy",
		action="store_true",
		help="Disable Webshare proxy usage for caption fetching (even if env vars set).",
	)
	parser.add_argument(
		"--rss-only",
		action="store_true",
		help="Discover today's uploads via channel RSS feeds (requires cached subscriptions) instead of YouTube Data API.",
	)
	parser.add_argument(
		"--fill-missing-summaries",
		action="store_true",
		help="Only process videos that already have a transcript entry but null/empty summary in the JSON.",
	)
	return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
	check_ffmpeg_availability()
	args = parse_args(argv)

	# Honor --no-proxy by setting a runtime global before any transcript attempts
	if args.no_proxy:
		global DISABLE_PROXIES
		DISABLE_PROXIES = True
	summarize_today_uploads(
		target_sentences=args.sentences,
		max_videos=args.max_videos,
		debug=args.debug,
		show_empty=args.show_empty,
		whisper_model=args.whisper_model,
		chunk_seconds=args.chunk_seconds,
		refresh_channels=args.refresh_channels,
		refresh_today=args.refresh_today,
		rss_only=args.rss_only,
		fill_missing=args.fill_missing_summaries,
	)


if __name__ == "__main__":
	try:
		main(sys.argv[1:])
	except KeyboardInterrupt:
		print("Interrupted by user")
