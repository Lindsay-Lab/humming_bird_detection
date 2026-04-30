import argparse
from pathlib import Path

import requests

from humming_bird_detection.project_paths import project_paths

parser = argparse.ArgumentParser(
    description="Download hummingbird images from iNaturalist."
)
parser.add_argument(
    "--output",
    type=Path,
    default=project_paths.data.raw / "Trochilidae",
    help="Output directory to save images",
)
parser.add_argument(
    "--taxon_name", type=str, default="Trochilidae", help="Taxon name to search for"
)
parser.add_argument(
    "--quality_grade",
    type=str,
    default="research",
    help="Quality grade of observations",
)
parser.add_argument(
    "--photo_license",
    type=str,
    default="CC0,CC-BY,CC-BY-NC",
    help="Photo license types",
)
parser.add_argument(
    "--per_page", type=int, default=100, help="Number of results per page"
)
parser.add_argument("--page", type=int, default=1, help="Page number to start from")

args, _unknown = parser.parse_known_args()


def download_image(url, output_path):
    """Download an image from url and write it to output_path.

    Raises requests.RequestException on network failure or non-200 response.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Warning: failed to download {url}: {e}")
        return
    with open(output_path, "wb") as f:
        f.write(response.content)


if __name__ == "__main__":
    # Create a directory to save the images
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set the search parameters
    params = {
        "taxon_name": args.taxon_name,
        # 'quality_grade': args.quality_grade,  # 'research' quality grade is not available for all users
        # 'photo_license': args.photo_license,
        "per_page": args.per_page,
        "page": args.page,
    }
    # Print the search parameters for debugging
    print("Search parameters:", params)

    # Fetch data from iNaturalist API
    while True:
        try:
            response = requests.get(
                "https://api.inaturalist.org/v1/observations",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching page {params['page']}: {e}")
            break

        data = response.json()

        if not data["results"]:
            print("No results found.")
            break

        for result in data["results"]:
            if "photos" in result and result["photos"]:
                photo_url = result["photos"][0]["url"].replace("square", "original")
                photo_id = result["id"]
                output_path = output_dir / f"{photo_id}.jpg"
                download_image(photo_url, output_path)

        params["page"] += 1

    print(f"Downloaded images to {output_dir}")
