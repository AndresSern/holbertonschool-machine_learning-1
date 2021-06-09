#!/usr/bin/env python3
"""
script that takes in input from the user
with the prompt Q: and prints A: as a response
"""
while True:
    txt = input("Q: ")
    if txt in ["exit", "quit", "goodbye", "bye"]:
        print("A: Goodbye")
        break
    print("A:")
