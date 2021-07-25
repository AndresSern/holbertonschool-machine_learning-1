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
    url = 'https://swapi-api.hbtn.io/api/people/'
    try:
        data = requests.get(url).json()
        while data['next']:

            for person in data['results']:
                planet = requests.get(person['homeworld']).json()
                planets.add(planet['name'])
            url = data['next']
            data = requests.get(url).json()
        return planets
    except Exception:
        return []
