#!/usr/bin/env python3

"""
script that prints the location of a specific user
"""
import sys
import requests
import time

if __name__ == "__main__":
    """
    If the status code is 403, print Reset in X min
    where X is the number of minutes from now and
    the value of X-Ratelimit-Reset
    """
    url = sys.argv[1]
    req = requests.get(url)
    data = req.json()
    if req .status_code == 404:
        print("Not found")
    elif req .status_code == 200:
        print(data["location"])
    elif req .status_code == 403:
        limit = req.headers["X-Ratelimit-Reset"]
        x = (int(limit) - int(time.time())) / 60
        print("Reset in {} min".format(int(x)))
