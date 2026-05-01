import json

with open("../protocol/schemas.json") as f:
    schemas = json.load(f)

with open("../protocol/types/hand.json") as f:
    hand = json.load(f)