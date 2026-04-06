#!/usr/bin/env python3
"""Affiche le prochain lancement SpaceX."""

import requests

LAUNCHES_URL = "https://api.spacexdata.com/v4/launches/upcoming"
ROCKETS_URL = "https://api.spacexdata.com/v4/rockets"
LAUNCHPADS_URL = "https://api.spacexdata.com/v4/launchpads"


def fetch(url):
    """Effectue une requête GET et retourne le JSON."""
    return requests.get(url).json()


def main():
    """Affiche les infos du prochain lancement SpaceX."""
    flights = fetch(LAUNCHES_URL)
    launch = min(flights, key=lambda f: f['date_unix'])

    rocket = fetch(f"{ROCKETS_URL}/{launch['rocket']}")
    launchpad = fetch(f"{LAUNCHPADS_URL}/{launch['launchpad']}")

    print(f"{launch['name']} ({launch['date_local']}) "
          f"{rocket['name']} - {launchpad['name']} ({launchpad['locality']})")


if __name__ == '__main__':
    main()
