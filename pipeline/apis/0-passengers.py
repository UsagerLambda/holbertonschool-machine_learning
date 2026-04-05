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
            passengers = ship['passengers']
            if passengers in ("unknown", "n/a"):
                continue
            passengers = passengers.replace(",", "")
            if int(passengers) >= passengerCount:
                ships.append(ship['name'])
        url = data['next']
    return ships
