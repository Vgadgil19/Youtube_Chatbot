
from urllib.parse import urlparse, parse_qs
import re

def extract_video_id(url):
    try:
        parsed_url = urlparse(url.strip())
        host = parsed_url.netloc.lower().split(":")[0]

        if host == "youtu.be" or host.endswith(".youtu.be"):
            video_id = parsed_url.path.strip("/").split("/")[0]

        elif (
            host == "youtube.com"
            or host.endswith(".youtube.com")
            or host == "youtube-nocookie.com"
            or host.endswith(".youtube-nocookie.com")
        ):
            if parsed_url.path == "/watch":
                video_id = parse_qs(parsed_url.query).get("v", [None])[0]

            elif re.match(r"^/(shorts|embed|live)/[^/]+", parsed_url.path):
                video_id = parsed_url.path.split("/")[2]

            else:
                return None
        else:
            return None

        if video_id and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            return video_id

        return None

    except (ValueError, IndexError):
        return None


# Call the function OUTSIDE its definition
url = "https://youtu.be/203-Zj0FQG4?si=tcGt_bhtqoYiDP9H"

result = extract_video_id(url)
print(result)


