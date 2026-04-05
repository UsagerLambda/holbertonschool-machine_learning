#!/usr/bin/env python3
"""Module pour récupérer les planètes d'origine des espèces sentientes."""

import requests


def sentientPlanets():
    """Retourne les noms des planètes d'origine des espèces sentientes."""
    worlds = []
    url = "https://swapi-api.hbtn.io/api/species/"
    while url is not None:
        response = requests.get(url)
        data = response.json()
        for specie in data['results']:
            if (specie['designation'] != "sentient"
                    and specie['classification'] != "sentient"):
                continue
            hw = specie['homeworld']
            if hw is not None and "unknown" not in hw:
                response2 = requests.get(hw)
                data2 = response2.json()
                worlds.append(data2["name"])
        url = data['next']
    return worlds
