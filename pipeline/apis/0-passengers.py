#!/usr/bin/env python3
"""Module pour récupérer les vaisseaux disponibles depuis l'API SWAPI."""

import requests


def availableShips(passengerCount):
    """Retourne les vaisseaux pouvant accueillir passengerCount passagers."""
    ships = []
    url = "https://swapi-api.hbtn.io/api/starships/"
    while url is not None:
        response = requests.get(url)
        data = response.json()
        for ship in data['results']:
            if ship['crew'] == "unknown":
                continue
            crew = ship['crew'].replace(",", "")
            if "-" in crew:
                crew = crew.split("-")[1]
            if int(crew) >= passengerCount:
                ships.append(ship['name'])
        url = data['next']
    return ships
