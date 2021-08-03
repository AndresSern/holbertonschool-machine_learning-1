#!/usr/bin/env python3
"""
Python script that provides some stats
about Nginx logs stored in MongoDB
"""
from pymongo import MongoClient


if __name__ == "__main__":
    """
    Args:
        Database: logs
        Collection: nginx
        -x logs:
            x is the number of documents in this collection
        Methods:
        -the number of documents with the method
            ["GET", "POST", "PUT", "PATCH", "DELETE"]

        -search:
            method=GET
            path=/status
    """
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs_collection = client.logs.nginx
    x = logs_collection.count_documents({})

    method = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    search = {"method": "GET", "path": "/status"}
    print('{} logs'.format(x))
    print('Methods:')
    for m in method:
        count_method = logs_collection.count_documents({"method": m})
        print('\tmethod {}: {}'.format(m, count_method))
    print("{} status check".format(logs_collection.count_documents(search)))
