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
    #  --------get all launches
    url = "https://api.spacexdata.com/v4/launches/"
    req1 = requests.get(url)
    data = req1.json()
    # get all roks
    roks = requests.get("https://api.spacexdata.com/v4/rockets/").json()
    roks_id = set()
    for i in roks:
        roks_id.add(i['id'])
    b = {}
    for i in roks_id:
        rok = requests.get("https://api.spacexdata.com/v4/rockets/" +
                           i).json()
        b[i] = rok['name']

    for item in data:
        # rok = requests.get("https://api.spacexdata.com/v4/rockets/" +
        #                   item['rocket']).json()
        # print(b[item['rocket']])

        if b[item['rocket']] in dct_nbr_rocket:
            dct_nbr_rocket[b[item['rocket']]] += 1
        else:
            dct_nbr_rocket[b[item['rocket']]] = 1

    for key, value in sorted(dct_nbr_rocket.items(),
                             key=lambda item: item[1], reverse=True):
        print("{}: {}".format(key, value))
