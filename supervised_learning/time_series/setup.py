#!/usr/bin/env python3
"""
Setup script to create virtual environment, install dependencies,
download Bitcoin historical data, and create required directories.
"""

import os
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path("data")
DATASET_DIR = Path("dataset")
MODEL_DIR = Path("model")
NORMAL_DIR = Path("normal")
PLOTS_DIR = Path("plots")
VENV_DIR = Path("venv")

FILES = {
    "bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv": "15A-rLSrfZ0td7muSrYHy0WX9ZqrMweES",
    "coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv": "16MgiuBfQKzXPoWFWi2w-LKJuZ7LgivpE",
}


def create_directories():
    """Create required project directories."""
    for d in [DATA_DIR, DATASET_DIR, MODEL_DIR, NORMAL_DIR, PLOTS_DIR]:
        d.mkdir(exist_ok=True)
        print(f"Created: {d}/")


def setup_venv():
    """Create virtual environment and install dependencies."""
    if not VENV_DIR.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        print(f"Created: {VENV_DIR}/")
    else:
        print(f"Virtual environment already exists: {VENV_DIR}/")

    # Determine pip path
    if sys.platform == "win32":
        pip_path = VENV_DIR / "Scripts" / "pip"
    else:
        pip_path = VENV_DIR / "bin" / "pip"

    print("Installing dependencies...")
    subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
    print("Dependencies installed.")


def download_from_gdrive(file_id, destination):
    """Download a file from Google Drive."""
    import requests

    url = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(url, params={"id": file_id}, stream=True)

    # Handle large file confirmation
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            response = session.get(
                url, params={"id": file_id, "confirm": value}, stream=True
            )
            break

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


def download_data():
    """Download all required datasets."""
    for filename, file_id in FILES.items():
        dest = DATA_DIR / filename
        if dest.exists():
            print(f"Already exists: {filename}")
            continue

        print(f"Downloading: {filename}...")
        download_from_gdrive(file_id, dest)
        print(f"Downloaded: {filename}")


def main():
    print("Setting up Bitcoin Time Series project...\n")

    create_directories()
    setup_venv()
    download_data()

    # Determine python path for instructions
    if sys.platform == "win32":
        python_cmd = f"{VENV_DIR}\\Scripts\\python"
        activate_cmd = f"{VENV_DIR}\\Scripts\\activate"
    else:
        python_cmd = f"{VENV_DIR}/bin/python"
        activate_cmd = f"source {VENV_DIR}/bin/activate"

    print(f"\nSetup complete. Run:")
    print(f"  {activate_cmd}")
    print(f"  python preprocess_data.py")
    print(f"  python forecast_btc.py")


if __name__ == "__main__":
    main()
