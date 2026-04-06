#!/usr/bin/env python3
"""Affiche la fréquence d'utilisation de chaque fusée SpaceX."""

import requests

LAUNCHES_URL = "https://api.spacexdata.com/v4/launches"
ROCKETS_URL = "https://api.spacexdata.com/v4/rockets"


def fetch(url):
    """Effectue une requête GET et retourne le JSON."""
    return requests.get(url).json()


def main():
    """Affiche le nombre de lancements par fusée, triés par fréquence."""
    launches = fetch(LAUNCHES_URL)
    rockets_data = fetch(ROCKETS_URL)
    rockets = {r["id"]: r["name"] for r in rockets_data}

    counts = {}

    for launch in launches:
        name = rockets[launch["rocket"]]
        counts[name] = counts.get(name, 0) + 1

    sorted_counts = sorted(
        counts.items(),
        key=lambda x: (-x[1], x[0])
    )

    for name, count in sorted_counts:
        print(f"{name}: {count}")


if __name__ == '__main__':
    main()
