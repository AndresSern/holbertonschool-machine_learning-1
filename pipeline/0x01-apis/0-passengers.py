#!/usr/bin/env python3
'''
a method that returns the list of ships
that can hold a given number of passengers
'''
import requests


def availableShips(passengerCount):
    """
    amethod that returns the list of ships that can hold
    a given number of passengers
        --ARGS: passengerCoun :{number of passengers}
        --Returns: list of ships
    """
    availableShips = []
    url = 'https://swapi-api.hbtn.io/api/starships/'
    try:
        data = requests.get(url).json()
        while data['next']:

            for ship in data['results']:
                psng = ship["passengers"]
                psng = psng.replace(',', '')

                if psng.isnumeric() and int(psng) >= passengerCount:
                    availableShips.append(ship['name'])
            url = data['next']
            data = requests.get(url).json()
        return availableShips
    except Exception:
        return []
