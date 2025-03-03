import os
import requests
from dotenv import load_dotenv

# load env
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")


def fetch_popular_movies(page=1):
    """
    Fetches a list of popular movies from TMDb for a given page.

    Args:
        page (int, optional): Page number to fetch. Defaults to 1.

    Returns:
        list: A list of movie dictionaries.
    """
    url = "https://api.themoviedb.org/3/movie/popular"
    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US",
        "page": page
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get("results", [])
    else:
        print(
            f"Error fetching popular movies (page {page}): {response.status_code}")
        return []


def fetch_movie_details(movie_id):
    """
    Fetches full movie details (including overview, tagline, etc.) using movie ID.

    Args:
        movie_id (int): The ID of the movie to fetch details for.

    Returns:
        dict: A dictionary containing movie details, or None if an error occurred.
    """
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(
            f"Error fetching details for movie ID {movie_id}: {response.status_code}")
        return None


def build_training_file(num_pages=5, output_file="movie_data.txt"):
    """
    Fetches movie data from TMDb (from multiple pages of popular movies)
    and writes it to a text file for fine-tuning.

    Args:
        num_pages (int, optional): Number of pages of popular movies to fetch. Defaults to 5.
        output_file (str, optional): The file to write the movie data to. Defaults to "movie_data.txt".
    """
    entries = []
    for page in range(1, num_pages + 1):
        movies = fetch_popular_movies(page)
        for movie in movies:
            movie_id = movie.get("id")
            details = fetch_movie_details(movie_id)
            if details:
                title = details.get("title", "")
                tagline = details.get("tagline", "")
                overview = details.get("overview", "")
                entry = f"Title: {title}\nTagline: {tagline}\nOverview: {overview}"
                entries.append(entry)
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry + "\n\n")

    print(
        f"Training file '{output_file}' created with {len(entries)} entries.")


if __name__ == "__main__":
    build_training_file(num_pages=5)
