#!/usr/bin/env python3
""" 31. Insert a document in Python"""


def insert_school(mongo_collection, **kwargs):
    """
    Args:
    mongo_collection will be the pymongo collection object
    Returns the new _id
    """
    row = mongo_collection.insert_one(kwargs)
    return row.inserted_id
