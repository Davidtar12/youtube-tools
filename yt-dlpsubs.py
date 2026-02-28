#!/usr/bin/env python3
"""Fetch and normalize YouTube subtitles using yt-dlp only.

Usage (examples):

  python yt-dlpsubs.py --url https://www.youtube.com/watch?v=VePxCcF99w4
  python yt-dlpsubs.py --url VePxCcF99w4 --langs en,en-US,es --best-only --save merged.txt
  python yt-dlpsubs.py --url VePxCcF99w4 --list-only

Features:
  * Accepts full URL or bare video ID.
  * Attempts to download manual subtitles first; also enables auto-generated subtitles (writeautomaticsub).
  * Restricts downloads to preferred language codes (but yt-dlp may fetch close variants).
  * Parses produced .vtt files, strips timing / sequence / WEBVTT headers, merges to plain text.
  * Can emit either best language only or all languages discovered.
  * Optional --save writes merged text to file; always echoes to stdout unless --quiet.

Exit codes:
  0 success (even if some languages missing)
  2 no subtitles were found / parsed
  3 invalid arguments

NOTE: This does NOT fall back to API scraping or Whisper transcription; pure yt-dlp path.
"""
from __future__ import annotations

import argparse
import re
import time
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from yt_dlp import YoutubeDL

def normalize_video_id(url_or_id: str) -> str:
	# Accept raw ID or full URL
	if "youtube.com" in url_or_id or "youtu.be" in url_or_id:
		# Simple extraction; yt-dlp could parse but we do quick regex
		import urllib.parse as _up
		try:
			if "youtu.be/" in url_or_id:
				return url_or_id.rsplit("/", 1)[-1].split("?")[0]
			parsed = _up.urlparse(url_or_id)
			qs = _up.parse_qs(parsed.query)
			if "v" in qs and qs["v"]:
				return qs["v"][0]
		except Exception:
			pass
	return url_or_id.strip()


def parse_webvtt(path: Path) -> str:
	lines_out: List[str] = []
	try:
		raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
	except Exception:
		return ""
	for line in raw:
		s = line.strip()
		if not s:
			continue
		if s.startswith("WEBVTT"):
			continue
		if "-->" in s:  # timing line
			continue
		if s.isdigit():  # sequence number
			continue
		# Remove position/style cues in trailing metadata (basic heuristic)
		s = re.sub(r"<[^>]+>", "", s)
		lines_out.append(s)
	text = "\n".join(lines_out).strip()
	return text


def fetch_subtitles(
	video_id: str,
	prefer_langs: List[str],
	tmp_path: Path,
	cookies_path: Path | None = None,
	write_all: bool = True,
	quiet: bool = False,
	per_lang_attempt: bool = False,
	max_retries: int = 2,
) -> Tuple[Dict[str, str], List[Path]]:
	"""Use yt-dlp to download subtitles (.vtt) for given video id.

	Returns mapping language->text and list of raw vtt paths.
	"""
	outtmpl = str(tmp_path / f"{video_id}.%(lang)s.%(ext)s")
	opts = {
		"skip_download": True,
		"writesubtitles": True,
		"writeautomaticsub": True,
		"subtitlesformat": "vtt",
		# Provide custom language list; yt-dlp will still fallback if needed
		"subtitleslangs": prefer_langs,
		"outtmpl": outtmpl,
		"quiet": True,
		"ignoreerrors": False,
		"no_warnings": True,
	}
	url = f"https://www.youtube.com/watch?v={video_id}"
	info = None
	# Basic retry logic for transient 429 (Too Many Requests) errors
	attempts = 2
	for attempt in range(1, attempts + 1):
		try:
			with YoutubeDL(opts) as ydl:
				# download=True is required to actually write subtitle files when skip_download is set
				info = ydl.extract_info(url, download=True)
			break
		except Exception as e:
			err_text = str(e)
			is_rate = '429' in err_text or 'Too Many Requests' in err_text
			if not quiet:
				print(f"yt-dlp metadata/subtitle extraction failed (attempt {attempt}/{attempts}): {e}", file=sys.stderr)
			if is_rate and attempt < attempts:
				# Short exponential backoff
				sleep_secs = 2 ** attempt
				if not quiet:
					print(f"[retry] Sleeping {sleep_secs}s due to 429 before retrying...", file=sys.stderr)
				time.sleep(sleep_secs)
				continue
			else:
				break

	# Collect downloaded .vtt files
	vtts = sorted(tmp_path.glob(f"{video_id}.*.vtt"))
	results: Dict[str, str] = {}
	for vtt_file in vtts:
		# Derive language code between video_id and extension (videoid.lang.vtt)
		parts = vtt_file.name.split('.')
		if len(parts) < 3:
			continue
		lang = parts[-2]
		text = parse_webvtt(vtt_file)
		if text and lang not in results:
			results[lang] = text

	# Fallback attempt: try all languages if none found
	if not results:
		if not quiet:
			print("[fallback] Re-attempting with subtitleslangs=all", file=sys.stderr)
		opts_all = dict(opts)
		opts_all['subtitleslangs'] = ['all']
		try:
			with YoutubeDL(opts_all) as ydl:
				info = ydl.extract_info(url, download=True)
		except Exception as e:
			if not quiet:
				print(f"[fallback] Attempt failed: {e}", file=sys.stderr)
		else:
			vtts = sorted(tmp_path.glob(f"{video_id}.*.vtt"))
			for vtt_file in vtts:
				parts = vtt_file.name.split('.')
				if len(parts) < 3:
					continue
				lang = parts[-2]
				text = parse_webvtt(vtt_file)
				if text and lang not in results:
					results[lang] = text

	# If nothing downloaded, but metadata has subtitles, provide a richer diagnostic
	if not results and info and not quiet:
		meta_subs = info.get('subtitles') or {}
		meta_auto = info.get('automatic_captions') or {}
		if meta_subs or meta_auto:
			avail_langs = sorted(set(list(meta_subs.keys()) + list(meta_auto.keys())))
			print(
				"[diagnostic] Subtitles metadata present but no .vtt files materialized. This can happen if: "
				"(1) the specified --langs excludes available tracks, (2) YouTube returned only formats yt-dlp didn't write, or "
				"(3) a temporary write issue occurred.",
				file=sys.stderr,
			)
			print(f"[diagnostic] Available (meta) languages: {', '.join(avail_langs)}", file=sys.stderr)
			print(f"[diagnostic] Preferred (--langs) order: {prefer_langs}", file=sys.stderr)
			if 'en' not in prefer_langs and 'en-US' in avail_langs:
				print("[hint] Consider adding 'en' to --langs (some tracks use plain 'en').", file=sys.stderr)
	elif not results and not quiet:
		print("[diagnostic] No subtitle metadata present at all (no manual or automatic captions).", file=sys.stderr)
	# If we only want languages supplied, we still collected only those due to subtitleslangs
	return results, vtts


def select_best(results: Dict[str, str], prefer_langs: List[str]) -> Tuple[str, str]:
	# Return (lang, text) best match by order of prefer_langs; else longest text
	for pl in prefer_langs:
		# match exact or case-insensitive
		for actual in results:
			if actual.lower() == pl.lower():
				return actual, results[actual]
	# fallback longest
	if results:
		lang = max(results.items(), key=lambda kv: len(kv[1]))[0]
		return lang, results[lang]
	return "", ""


def build_arg_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Fetch YouTube subtitles via yt-dlp only and output plain text.")
	p.add_argument('--url', required=True, help='YouTube video URL or ID')
	p.add_argument('--langs', default='en,en-US,en-GB,es,es-419', help='Comma separated preferred language codes order.')
	p.add_argument('--all-langs', action='store_true', help='Print all fetched languages (with headers) instead of only best match.')
	p.add_argument('--best-only', action='store_true', help='Force only best single language output (default if --all-langs not set).')
	p.add_argument('--save', help='Write merged/best transcript to this file.')
	p.add_argument('--keep-vtt', action='store_true', help='Keep downloaded .vtt files (they are deleted by default).')
	p.add_argument('--list-only', action='store_true', help='Only list available subtitle languages and exit.')
	p.add_argument('--quiet', action='store_true', help='Suppress informational stderr messages.')
	return p


def main(argv: List[str] | None = None) -> int:
	args = build_arg_parser().parse_args(argv)
	video_id = normalize_video_id(args.url)
	prefer_langs = [c.strip() for c in args.langs.split(',') if c.strip()]
	if not prefer_langs:
		if not args.quiet:
			print("No languages specified after parsing --langs", file=sys.stderr)
		return 3
	with tempfile.TemporaryDirectory(prefix="yt_subs_") as tmp:
		tmp_path = Path(tmp)
		results, vtt_paths = fetch_subtitles(video_id, prefer_langs, tmp_path, quiet=args.quiet)
		if args.list_only:
			if results:
				print("Available subtitle languages:")
				for lang in sorted(results.keys()):
					print(f"  {lang} (chars={len(results[lang])})")
				return 0
			else:
				print("No subtitles available.")
				return 2
		if not results:
			if not args.quiet:
				print("No subtitles fetched (none available or download failed).", file=sys.stderr)
			return 2
		if args.all_langs and not args.best_only:
			# Print each language with a separator banner
			merged_out = []
			for lang, text in sorted(results.items()):
				merged_out.append(f"===== {lang} =====\n{text}\n")
			final_text = "\n".join(merged_out).strip()
		else:
			lang, text = select_best(results, prefer_langs)
			final_text = text
			if not args.quiet:
				print(f"[info] Selected language: {lang or 'unknown'} (chars={len(text)})", file=sys.stderr)
		if args.save:
			try:
				Path(args.save).write_text(final_text, encoding='utf-8')
				if not args.quiet:
					print(f"[info] Wrote transcript to {args.save}", file=sys.stderr)
			except Exception as e:
				if not args.quiet:
					print(f"[warn] Failed to write {args.save}: {e}", file=sys.stderr)
		# Always print to stdout unless suppressed (we still print even if --save for pipeline usage)
		print(final_text)
		# Optionally persist vtt files by copying to CWD
		if args.keep_vtt:
			for v in vtt_paths:
				dest = Path.cwd() / v.name
				try:
					dest.write_text(v.read_text(encoding='utf-8'), encoding='utf-8')
				except Exception:
					pass
	return 0


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True, help='YouTube video URL or ID')
    parser.add_argument('--list-only', action='store_true', help='List available subtitles without downloading')
    parser.add_argument('--cookies', type=str, default=None, help='Path to cookies file')
    parser.add_argument('--quiet', action='store_true', help='Suppress non-error output')
    parser.add_argument('--per-lang-attempt', action='store_true', help='Attempt each language separately on failure')
    parser.add_argument('--max-retries', type=int, default=2, help='Maximum number of retries for transient errors')
    parser.add_argument('--langs', nargs='+', default=[], help='Preferred subtitle languages')
    
    args = parser.parse_args()
    
    video_id = normalize_video_id(args.url)
    prefer_langs = args.langs
    tmp_path = Path("temp_subs.vtt")
    cookies_path = Path(args.cookies) if args.cookies else None
    
    fetch_subtitles(
        video_id=video_id,
        prefer_langs=prefer_langs,
        tmp_path=tmp_path,
        cookies_path=cookies_path,
        quiet=args.quiet,
        per_lang_attempt=args.per_lang_attempt,
        max_retries=args.max_retries
    )


if __name__ == '__main__':
	try:
		raise SystemExit(main())
	except KeyboardInterrupt:
		print("Interrupted", file=sys.stderr)
		raise SystemExit(130)

