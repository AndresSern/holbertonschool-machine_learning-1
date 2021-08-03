#!/usr/bin/env python3
"""function that returns the list of school having a specific topic"""


def schools_by_topic(mongo_collection, topic):
    """
    Args:
        -mongo_collection:{pymongo collection object}
        -topic: (string) will be topic searched
    """
    list_search = []
    data = mongo_collection.find({'topics': {'$all': [topic]}})
    for item in data:
        list_search.append(item)
    return list_search
