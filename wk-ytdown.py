import subprocess

def download_video(video_url):
    # Specify 720p video quality
    command = ['yt-dlp', '-f', 'bestvideo[height<=720]+bestaudio/best[height<=720]', video_url]
    try:
        subprocess.run(command, check=True)
        print(f"Video downloaded successfully: {video_url}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download video: {video_url}")
        print(f"Error: {e}")

# Example usage
video_url = 'https://www.youtube.com/watch?v=lxuTCHJSers'  # This is just an example URL
download_video(video_url)