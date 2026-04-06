#!/usr/bin/env python3
"""Module pour récupérer la localisation d'un utilisateur GitHub via l'API."""

import sys
import requests
import time


def main():
    """Récupère la localisation d'un utilisateur GitHub via l'API."""
    now = int(time.time())
    args = sys.argv[1]
    response = requests.get(args)
    if response.status_code == 200:
        data = response.json()
        print(data['location'])
    elif response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        data = response.headers
        X_rate = int(data['X-Ratelimit-Reset'])
        X_rate = (X_rate - now) // 60
        print(f"Reset in {X_rate} min")


if __name__ == '__main__':
    main()
