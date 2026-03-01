# youtube-tools

YouTube automation: transcript merger, AI chapter generator (Whisper + Gemini), playlist downloader, AI summarizer.

## Scripts

| Script | Description |
|--------|-------------|
| `chaptermaker.py` | Auto-generates YouTube chapter timestamps from a video — transcribes with Whisper, then uses Gemini AI to identify topic breaks. **Written manually.** |
| `youtube_transcript_merger.py` | Downloads and merges transcripts from multiple YouTube videos into a single document |
| `wk-youtube-summ.py` | Downloads a video transcript and summarises it using Gemini AI |
| `yt-summ-sin-whisper.py` | Summarises YouTube videos using transcript API (no local Whisper required) |
| `youtube_description.py` | Generates an SEO-optimised YouTube video description using AI given a title and topic |
| `youtube_playlist_downloader.py` | Downloads all videos from a YouTube playlist via yt-dlp |
| `yt-dlpsubs.py` | Downloads videos with embedded subtitles/captions using yt-dlp |
| `pruebyttranscriptapi.py` | Test/debug script for the YouTube Transcript API |

## Prerequisites

- Python 3.9+
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) installed: `pip install yt-dlp`
- FFmpeg (for video processing): `ffmpeg -version`
- Gemini API key or OpenAI API key for summarisation

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GEMINI_API_KEY, YOUTUBE_API_KEY
```

## Usage

```bash
# Generate chapter timestamps for a video
python chaptermaker.py --url "https://youtube.com/watch?v=VIDEO_ID"

# Summarise a YouTube video
python wk-youtube-summ.py --url "https://youtube.com/watch?v=VIDEO_ID"

# Download a full playlist
python youtube_playlist_downloader.py --url "https://youtube.com/playlist?list=PLAYLIST_ID"

# Merge transcripts from multiple videos
python youtube_transcript_merger.py --urls urls.txt --output merged.txt
```

## Built with

Python · OpenAI Whisper · Gemini API · yt-dlp · YouTube Transcript API  
AI-assisted development (Claude, GitHub Copilot) — architecture, requirements, QA validation and debugging by me. `chaptermaker.py` written manually.
