#!/usr/bin/env python3
"""
Script to download files from a private Figshare item.

Usage:
    python download_figshare.py --article_id <id> --token <token> [--output_dir <dir>]

Note: You need a Figshare API token with read access to private items.
    Get your token from https://figshare.com/account/applications
"""

import argparse
from pathlib import Path

import requests


def download_figshare_files(article_id: str, token: str, output_dir: str = "."):
    """
    Download all files from a private Figshare article.

    Parameters
    ----------
    article_id : str
        The Figshare article ID
    token : str
        Your Figshare API token
    output_dir : str
        Directory to save files (default: current directory)
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # API headers
    headers = {"Authorization": f"token {token}", "Content-Type": "application/json"}

    # API URLs
    files_url = f"https://api.figshare.com/v2/account/articles/{article_id}/files"

    # Get file list with pagination
    files = []
    page = 1
    page_size = 100  # Max allowed by Figshare API

    while True:
        paginated_url = f"{files_url}?page={page}&page_size={page_size}"
        print(f"Fetching page {page} from {paginated_url}")

        try:
            response = requests.get(paginated_url, headers=headers)
            response.raise_for_status()
            page_files = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            break

        if not page_files:
            break

        files.extend(page_files)
        page += 1

        # Safety check to avoid infinite loop
        if page > 100:  # Arbitrary large number
            print("Too many pages, stopping to prevent infinite loop")
            break

    if not files:
        print("No files found in this article.")
        return

    print(f"Found {len(files)} files. Starting download...")

    # Download each file
    for file_info in files:
        file_id = file_info["id"]
        filename = file_info["name"]
        size_mb = file_info.get("size", 0) / (1024 * 1024)

        print(f"Downloading {filename} ({size_mb:.1f} MB)...")

        # Get download URL
        download_url = (
            f"https://api.figshare.com/v2/account/articles/{article_id}/files/{file_id}"
        )
        try:
            response = requests.get(download_url, headers=headers)
            response.raise_for_status()
            download_info = response.json()
            actual_download_url = download_info["download_url"]
        except requests.exceptions.RequestException as e:
            print(f"Error getting download URL for {filename}: {e}")
            continue

        # Download file
        try:
            with requests.get(actual_download_url, headers=headers, stream=True) as r:
                r.raise_for_status()
                file_path = output_path / filename
                with open(file_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Downloaded {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")

    print("Download complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Download files from private Figshare item"
    )
    parser.add_argument("--article_id", required=True, help="Figshare article ID")
    parser.add_argument("--token", required=True, help="Figshare API token")
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Output directory (default: current directory)",
    )

    args = parser.parse_args()

    download_figshare_files(args.article_id, args.token, args.output_dir)


if __name__ == "__main__":
    main()
