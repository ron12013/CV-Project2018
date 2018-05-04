import json
from pprint import pprint
data = json.load(open('vis.json'))

pprint(data[0][0])
