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

    data = requests.get(url).json()
    while data['next']:

        for species in data['results']:
            if species['designation'] == 'sentient'\
                 and species['homeworld'] is not None:
                planet = requests.get(species['homeworld']).json()
                planets.add(planet['name'])
        url = data['next']
        data = requests.get(url).json()
    return planets
