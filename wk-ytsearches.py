from dotenv import load_dotenv
import os
load_dotenv()

# Assuming you have installed Google's API client library
# You can install it via pip: pip install --upgrade google-api-python-client
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Placeholder for your YouTube Data API key
# You need to replace 'YOUR_API_KEY' with your actual YouTube Data API key
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

def youtube_search(query, max_results=1):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                    developerKey=YOUTUBE_API_KEY)

    search_response = youtube.search().list(
        q=query,
        part='id,snippet',
        maxResults=max_results,
        type='video',
        videoDuration='long'  # This parameter is for videos longer than 20 minutes
    ).execute()

    videos = []

    for search_result in search_response.get('items', []):
        if search_result['id']['kind'] == 'youtube#video':
            videos.append(f"https://www.youtube.com/watch?v={search_result['id']['videoId']}")

    return videos

# Multi-line string of queries, one per line, easy to copy-paste and modify.
query_string = """
Turkey 4K
Greece 4K
Jordan 4K
Thailand 4k
Indonesia 4k
Philippines 4k
Nigeria 4k
Morocco 4k
Namibia 4k
Senegal 4k
Zambia 4k
South Africa 4k
Japan 4k
Korea 4k
Vietnam 4k
Cambodia 4k
"""

# Split the multi-line string into a list of queries.
queries = [line.strip() for line in query_string.strip().split('\n')]

for query in queries:
    try:
        video_urls = youtube_search(query + " 4k", max_results=1)
        if video_urls:
            print(f"Query: {query}, URL: {video_urls[0]}")
        else:
            print(f"Query: {query}, No suitable video found")
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
