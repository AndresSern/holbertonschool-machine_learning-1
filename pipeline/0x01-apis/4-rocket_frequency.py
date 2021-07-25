#!/usr/bin/env python3


""" script that displays the number of launches per rocket."""

import requests

if __name__ == '__main__':
    ''''
    name_set = set()
    rock_url = "https://api.spacexdata.com/v4/rockets/"
    req3 = requests.get(rock_url)
    rock_data = req3.json()
    for item in rock_data:
        name_set.add(item['name'])
    dct_nbr_rocket = {i: 0 for i in name_set}
    '''
    dct_nbr_rocket = {}
    #  ---------------
    url = "https://api.spacexdata.com/v4/launches/"
    req1 = requests.get(url)
    data = req1.json()
    for item in data:
        rok = requests.get("https://api.spacexdata.com/v4/rockets/" +
                           item['rocket']).json()


        if rok['name'] in dct_nbr_rocket.keys():
            dct_nbr_rocket[rok['name']] += 1
        else:
            dct_nbr_rocket[rok['name']] = 1


    result = {k: v for k, v in sorted(dct_nbr_rocket.items(),
                                      key=lambda item: item[1], reverse=True)}
    result = {key: val for key, val in result.items() if val != 0}
    for key, value in result.items():
        print(key, ': ', value)
