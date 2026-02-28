#!/usr/bin/env python3
"""YouTube Playlist Downloader with Audio Enhancement

Downloads YouTube playlists as audio files with professional audio enhancement.
Uses Webshare proxy, preserves playlist order, and enhances audio quality.

Features:
- Downloads entire playlists maintaining original order
- Applies noise reduction (noisereduce)
- Applies compression, EQ, and limiting (Spotify's pedalboard)
- Normalizes loudness to broadcast standards
- Outputs high-quality MP3 files

Usage:
    python youtube_playlist_downloader.py "https://www.youtube.com/playlist?list=..."
    python youtube_playlist_downloader.py "https://www.youtube.com/playlist?list=..." --output "downloads\MyPlaylist"

Requirements:
    pip install yt-dlp pedalboard noisereduce pydub scipy numpy python-dotenv
    FFmpeg must be in PATH
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
from dotenv import load_dotenv
from scipy.io import wavfile

# Load environment variables
load_dotenv("C:\\your_project_folder\\.env")

try:
    from yt_dlp import YoutubeDL
except ImportError:
    raise SystemExit("yt-dlp is required. Install with 'pip install yt-dlp'.")

try:
    import noisereduce as nr
except ImportError:
    nr = None
    print("Warning: noisereduce not installed. Noise reduction will be skipped.")

try:
    from pedalboard import Pedalboard, Compressor, Gain, LowShelfFilter, HighShelfFilter, Limiter
    from pedalboard.io import AudioFile
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False
    print("Warning: pedalboard not installed. Advanced audio processing will be skipped.")

try:
    from pydub import AudioSegment, effects as pydub_effects
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not installed. Volume normalization will be skipped.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Proxy configuration
# Set USE_PROXY=false to disable proxy (e.g., when Webshare bandwidth is exhausted)
USE_PROXY = os.getenv("USE_PROXY", "true").lower() == "true"
WEBSHARE_PROXY_URL = os.getenv("WEBSHARE_PROXY_URL") if USE_PROXY else None
if USE_PROXY and not WEBSHARE_PROXY_URL:
    logger.warning("WEBSHARE_PROXY_URL not set. Downloads may fail without proxy.")
elif not USE_PROXY:
    logger.info("Proxy disabled (USE_PROXY=false). Downloading directly.")

# Default output directory
DEFAULT_OUTPUT_ROOT = Path("downloads")


@dataclass
class TrackInfo:
    """Information about a single track in the playlist."""
    index: int  # 1-based index in playlist
    video_id: str
    title: str
    url: str
    duration: int  # seconds


def sanitize_filename(name: str, max_length: int = 100) -> str:
    """Sanitize a string to be safe for use as a filename."""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '')
    # Replace multiple spaces with single space
    name = re.sub(r'\s+', ' ', name).strip()
    # Truncate if too long
    if len(name) > max_length:
        name = name[:max_length].rsplit(' ', 1)[0]
    return name


def get_playlist_info(playlist_url: str) -> Tuple[str, List[TrackInfo]]:
    """Extract playlist metadata and track list.
    
    Returns:
        Tuple of (playlist_title, list of TrackInfo)
    """
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,  # Don't download, just get info
        "skip_download": True,
    }
    
    if WEBSHARE_PROXY_URL:
        ydl_opts["proxy"] = WEBSHARE_PROXY_URL
    
    logger.info(f"Fetching playlist info: {playlist_url}")
    
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
    
    playlist_title = info.get("title", "Unknown Playlist")
    entries = info.get("entries", [])
    
    tracks = []
    for idx, entry in enumerate(entries, start=1):
        if entry is None:
            continue
        tracks.append(TrackInfo(
            index=idx,
            video_id=entry.get("id", ""),
            title=entry.get("title", f"Track {idx}"),
            url=entry.get("url") or f"https://www.youtube.com/watch?v={entry.get('id', '')}",
            duration=entry.get("duration", 0) or 0,
        ))
    
    logger.info(f"Found {len(tracks)} tracks in playlist: {playlist_title}")
    return playlist_title, tracks


def download_audio(track: TrackInfo, output_dir: Path, temp_dir: Path) -> Optional[Path]:
    """Download audio for a single track.
    
    Returns path to downloaded WAV file, or None if failed.
    """
    # Create numbered filename to preserve order
    safe_title = sanitize_filename(track.title)
    output_template = str(temp_dir / f"{track.index:03d}_%(id)s.%(ext)s")
    
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "quiet": True,
        "no_warnings": True,
    }
    
    if WEBSHARE_PROXY_URL:
        ydl_opts["proxy"] = WEBSHARE_PROXY_URL
    
    try:
        logger.info(f"[{track.index}] Downloading: {track.title}")
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(track.url, download=True)
            downloaded = Path(ydl.prepare_filename(info))
        
        # yt-dlp adds .wav extension during post-processing
        wav_path = downloaded.with_suffix(".wav")
        if wav_path.exists():
            return wav_path
        
        # Fallback: look for any matching file
        pattern = f"{track.index:03d}_*.wav"
        matches = list(temp_dir.glob(pattern))
        if matches:
            return matches[0]
        
        logger.error(f"[{track.index}] Downloaded file not found")
        return None
        
    except Exception as e:
        logger.error(f"[{track.index}] Download failed: {e}")
        return None


def enhance_audio_pedalboard(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply professional audio enhancement using Spotify's Pedalboard.
    
    Applies:
    - Low shelf boost for warmth
    - High shelf for clarity
    - Compression to even out dynamics
    - Limiting to prevent clipping
    """
    if not PEDALBOARD_AVAILABLE:
        return audio_data
    
    # Ensure float32 format for pedalboard
    if audio_data.dtype != np.float32:
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        else:
            audio_data = audio_data.astype(np.float32)
    
    # Ensure 2D array (samples, channels) - pedalboard expects (channels, samples)
    if audio_data.ndim == 1:
        audio_data = audio_data.reshape(1, -1)  # Mono: (1, samples)
    elif audio_data.ndim == 2 and audio_data.shape[0] > audio_data.shape[1]:
        # Shape is (samples, channels), transpose to (channels, samples)
        audio_data = audio_data.T
    
    # Build the effects chain for voice/audiobook enhancement
    board = Pedalboard([
        # Subtle low-end warmth
        LowShelfFilter(cutoff_frequency_hz=200, gain_db=2.0),
        # High-end clarity for voice
        HighShelfFilter(cutoff_frequency_hz=3000, gain_db=1.5),
        # Gentle compression to even out dynamics (good for audiobooks)
        Compressor(
            threshold_db=-20,
            ratio=3.0,
            attack_ms=10,
            release_ms=100,
        ),
        # Makeup gain
        Gain(gain_db=3.0),
        # Limiter to prevent clipping
        Limiter(threshold_db=-1.0, release_ms=100),
    ])
    
    # Process audio
    enhanced = board(audio_data, sample_rate)
    
    # Transpose back if needed (samples, channels)
    if enhanced.ndim == 2:
        enhanced = enhanced.T
    
    return enhanced


def enhance_audio(input_path: Path, output_path: Path) -> bool:
    """Apply full audio enhancement pipeline.
    
    1. Noise reduction (noisereduce)
    2. EQ, compression, limiting (pedalboard)
    3. Loudness normalization (pydub)
    
    Returns True if successful.
    """
    try:
        logger.info(f"  Enhancing audio...")
        
        # Load audio
        sample_rate, audio_data = wavfile.read(str(input_path))
        original_dtype = audio_data.dtype
        
        # Convert to float for processing
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_float = audio_data.astype(np.float32) / 2147483648.0
        else:
            audio_float = audio_data.astype(np.float32)
        
        # Step 1: Noise reduction
        if nr is not None:
            logger.debug("  Applying noise reduction...")
            # Handle mono vs stereo
            if audio_float.ndim == 1:
                audio_float = nr.reduce_noise(
                    y=audio_float,
                    sr=sample_rate,
                    prop_decrease=0.7,  # Keep some natural room tone
                    stationary=True,
                )
            else:
                # Process each channel
                channels = []
                for ch in range(audio_float.shape[1]):
                    reduced = nr.reduce_noise(
                        y=audio_float[:, ch],
                        sr=sample_rate,
                        prop_decrease=0.7,
                        stationary=True,
                    )
                    channels.append(reduced)
                audio_float = np.column_stack(channels)
        
        # Step 2: Pedalboard enhancement
        if PEDALBOARD_AVAILABLE:
            logger.debug("  Applying pedalboard effects...")
            audio_float = enhance_audio_pedalboard(audio_float, sample_rate)
        
        # Ensure we have proper shape
        if audio_float.ndim == 1:
            audio_float = audio_float.reshape(-1, 1)
        elif audio_float.ndim == 2 and audio_float.shape[0] < audio_float.shape[1]:
            audio_float = audio_float.T
        
        # Convert back to int16 for saving
        audio_int16 = np.clip(audio_float * 32768.0, -32768, 32767).astype(np.int16)
        
        # Save intermediate WAV
        temp_wav = output_path.with_suffix('.enhanced.wav')
        wavfile.write(str(temp_wav), sample_rate, audio_int16)
        
        # Step 3: Final normalization with pydub and convert to MP3
        if PYDUB_AVAILABLE:
            logger.debug("  Normalizing and converting to MP3...")
            audio = AudioSegment.from_wav(str(temp_wav))
            
            # Normalize to -1 dBFS (broadcast standard)
            normalized = pydub_effects.normalize(audio, headroom=1.0)
            
            # Export as high-quality MP3
            normalized.export(
                str(output_path),
                format="mp3",
                bitrate="192k",
                parameters=["-q:a", "2"],  # High quality VBR
            )
            
            # Cleanup temp file
            temp_wav.unlink(missing_ok=True)
        else:
            # Just convert to MP3 with ffmpeg
            subprocess.run([
                "ffmpeg", "-y", "-i", str(temp_wav),
                "-codec:a", "libmp3lame", "-b:a", "192k",
                str(output_path)
            ], capture_output=True, check=True)
            temp_wav.unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        logger.error(f"  Enhancement failed: {e}")
        return False


def process_playlist(
    playlist_url: str,
    output_dir: Optional[Path] = None,
    skip_existing: bool = True,
) -> int:
    """Download and enhance an entire playlist.
    
    Args:
        playlist_url: YouTube playlist URL
        output_dir: Output directory (auto-created from playlist title if None)
        skip_existing: Skip tracks that already exist
        
    Returns:
        Number of successfully processed tracks
    """
    # Get playlist info
    playlist_title, tracks = get_playlist_info(playlist_url)
    
    if not tracks:
        logger.error("No tracks found in playlist")
        return 0
    
    # Create output directory
    if output_dir is None:
        safe_title = sanitize_filename(playlist_title)
        output_dir = DEFAULT_OUTPUT_ROOT / safe_title
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Create temp directory for downloads
    temp_dir = Path(tempfile.mkdtemp(prefix="yt_playlist_"))
    
    success_count = 0
    
    try:
        for track in tracks:
            # Build output filename with track number prefix
            safe_title = sanitize_filename(track.title)
            output_file = output_dir / f"{track.index:03d} - {safe_title}.mp3"
            
            # Skip if exists
            if skip_existing and output_file.exists():
                logger.info(f"[{track.index}/{len(tracks)}] Skipping (exists): {track.title}")
                success_count += 1
                continue
            
            logger.info(f"[{track.index}/{len(tracks)}] Processing: {track.title}")
            
            # Download
            wav_path = download_audio(track, output_dir, temp_dir)
            if wav_path is None:
                continue
            
            # Enhance and convert
            if enhance_audio(wav_path, output_file):
                success_count += 1
                logger.info(f"[{track.index}/{len(tracks)}] ✓ Complete: {output_file.name}")
            
            # Cleanup downloaded WAV
            wav_path.unlink(missing_ok=True)
    
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {success_count}/{len(tracks)} tracks processed")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'='*60}")
    
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube playlists with audio enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "playlist_url",
        help="YouTube playlist URL",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help=f"Output directory (default: auto-created in {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-download and re-process existing files",
    )
    
    args = parser.parse_args()
    
    # Validate URL
    if "playlist" not in args.playlist_url.lower():
        logger.error("URL does not appear to be a playlist. Please provide a playlist URL.")
        sys.exit(1)
    
    # Process
    success = process_playlist(
        args.playlist_url,
        output_dir=args.output,
        skip_existing=not args.no_skip,
    )
    
    sys.exit(0 if success > 0 else 1)


if __name__ == "__main__":
    main()
