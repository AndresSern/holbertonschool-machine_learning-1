#!/usr/bin/env python3
'''
Where I am?
'''
import requests


def sentientPlanets():
    """
    method that returns the list of names of
    the home planets of all sentient species
    """
    planets = set()
    url = 'https://swapi-api.hbtn.io/api/species'

    try:
        while True:
            data = requests.get(url).json()
            for species in data['results']:
                if species["designation"] == "sentient" or\
                     species['classification'] == 'sentient':
                    if species['homeworld']:
                        planet = requests.get(species['homeworld']).json()
                        planets.add(planet['name'])

            url = data['next']
            data = requests.get(url).json()
    except Exception:
        return planets
