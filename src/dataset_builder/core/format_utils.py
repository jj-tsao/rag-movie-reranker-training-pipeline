from typing import Dict

def extract_media_data(media_type: str, media: Dict) -> Dict:
    # Common fields with type-specific fallbacks
    media_id = media.get("id", 0)
    title = media.get("title" if media_type == "movie" else "name", "Unknown")
    genre_list = [genre["name"] for genre in media.get("genres", [])]
    overview = media.get("overview", "No overview available.")
    tagline = media.get("tagline", "")
    star_list = media.get("stars", [])
    date_field = "release_date" if media_type == "movie" else "first_air_date"
    release_date = media.get(date_field, "")
    keyword_list = media.get("keywords", [])

    # Media type specific fields
    if media_type == "movie":
        director = media.get("director", "Unknown")
        collection = media.get("belongs_to_collection", {}).get("name", "") if media.get("belongs_to_collection") else ""
        specific_fields = {
            "collection": collection,
            "director": director,
        }
    else:  # TV show
        creator_list = media.get("creator", [])
        season_count = media.get('number_of_seasons', None)
        specific_fields = {
            "season_count": season_count,
            "creator": creator_list,
        }

    # Build metadata dictionary
    metadata = {
        "media_id": media_id,
        "media_type": media_type,
        "title": title,
        "genres": genre_list,
        "overview": overview,
        "tagline": tagline,
        "stars": star_list,
        "release_date": release_date,
        "keywords": keyword_list,
    }

    # Add media-specific fields to metadata
    metadata.update(specific_fields)

    return metadata


def format_training_text(media_type: str, media: Dict) -> str:
    title = media.get("title" if media_type == "movie" else "name", "Unknown") or "Unknown"
    genres = [genre["name"] for genre in media.get("genres", [""])]
    overview = media.get("overview", "").strip()
    tagline = media.get("tagline", "").strip()
    stars = media.get("stars", [])
    keywords = media.get("keywords", [])

    date_field = "release_date" if media_type == "movie" else "first_air_date"
    release_date = media.get(date_field, "")

    if media_type == "movie":
        director = media.get("director", "")
        specific_content = f"Director: {director}" if director else []
    else:  # TV show
        creator = media.get("creator", [])
        specific_content = f"Creator: {', '.join(creator)}" if creator else []

    parts = [
        f"Title: {title}",
        f"Genres: {', '.join(genres)}" if genres else [],
        f"Overview: {overview}" if overview else [],
        f"Tagline: {tagline}" if tagline else [],
        specific_content or [],
        f"Stars: {', '.join(stars)}" if stars else [],
        f"Release Date: {release_date[:10]}" if release_date else [],
        f"Keywords: {', '.join(keywords)}" if keywords else [],
    ]

    return "\n".join([part for part in parts if part]).strip()