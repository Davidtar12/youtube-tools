from youtube_transcript_api import YouTubeTranscriptApi
from pathlib import Path
import re

video_id = "piAFRoIhwZU"
ytt_api = YouTubeTranscriptApi()
fetched = ytt_api.fetch(video_id)

# Merge snippet-style transcripts into a single unified string
def snippets_to_merged_text(fetched_obj):
    """Return a single merged transcript string from a FetchedTranscript or list-like.

    Handles both new FetchedTranscript objects (with .to_raw_data()) and older list/dict formats.
    """
    # Try to get raw list-of-dicts if available
    raw = None
    try:
        raw = fetched_obj.to_raw_data()
    except Exception:
        raw = None

    parts = []
    if raw:
        for item in raw:
            if isinstance(item, dict):
                txt = item.get("text")
            else:
                txt = getattr(item, "text", None)
            if txt:
                parts.append(txt.strip())
    else:
        # Fall back: iterate fetched_obj which is iterable
        for item in fetched_obj:
            txt = getattr(item, "text", None) if not isinstance(item, dict) else item.get("text")
            if txt:
                parts.append(txt.strip())

    merged = " ".join(parts)
    # Normalize whitespace
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged

merged = snippets_to_merged_text(fetched)

# Print a short excerpt to the console so it's visible in script runs
excerpt_len = 2000
print("=== Merged transcript excerpt ===")
print(merged[:excerpt_len])
print("=== End excerpt ===")

# Write the full merged transcript to a file next to the script
out_path = Path(__file__).resolve().parent / f"{video_id}_merged.txt"
try:
    out_path.write_text(merged, encoding="utf-8")
    print(f"Wrote merged transcript to: {out_path}")
except Exception as e:
    print(f"Failed to write merged transcript to {out_path}: {e}")