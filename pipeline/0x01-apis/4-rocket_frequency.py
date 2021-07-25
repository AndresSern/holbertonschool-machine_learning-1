#!/usr/bin/env python3


"""  script that displays the upcoming launch """

import requests

a = set()
rock_url = "https://api.spacexdata.com/v4/rockets/"
req3 = requests.get(rock_url)
rock_data = req3.json()
for item in rock_data:
    a.add(item['name'])
dictOfWords = {i: 0 for i in a}

#  ---------------
url = "https://api.spacexdata.com/v4/launches/"
req1 = requests.get(url)
data = req1.json()
for item in data:
    rok = requests.get("https://api.spacexdata.com/v4/rockets/" + item['rocket']).json()

    try:
        dictOfWords[rok['name']] += 1
    except Exception:
        pass

result = {k: v for k, v in sorted(dictOfWords.items(),
                                  key=lambda item: item[1])}
for key, value in result.items():
    print(key, ' : ', value)
