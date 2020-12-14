# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:23:15 2020

@author: Ruben
"""


#%% Initialization and constants

import json
import requests
from requests.auth import HTTPBasicAuth

api_key = '25bafcdc29b94cc488b89c2ca9539a98'


# Functions to create: function to select 1km by 1km area around ship location
# Function to set all filters
# Function to give in AIS data
# Function to determine if satelite passes at time
# Find filter to make sure 100% of area is included



#%% Determining area and filters

# Area in lat,lon (created via geojson.io) 
geojson_geometry = {
  "type": "Polygon",
  "coordinates": [
    [ 
      [-121.59290313720705, 37.93444993515032],
      [-121.27017974853516, 37.93444993515032],
      [-121.27017974853516, 38.065932950547484],
      [-121.59290313720705, 38.065932950547484],
      [-121.59290313720705, 37.93444993515032]
    ]
  ]
}

# get images that overlap with our AOI 
geometry_filter = {
  "type": "GeometryFilter",
  "field_name": "geometry",
  "config": geojson_geometry
}

# get images acquired within a date range
date_range_filter = {
  "type": "DateRangeFilter",
  "field_name": "acquired",
  "config": {
    "gte": "2016-08-31T00:00:00.000Z",
    "lte": "2016-09-01T00:00:00.000Z"
  }
}

# only get images which have <50% cloud coverage
cloud_cover_filter = {
  "type": "RangeFilter",
  "field_name": "cloud_cover",
  "config": {
    "lte": 0.5
  }
}

# combine our geo, date, cloud filters
combined_filter = {
  "type": "AndFilter",
  "config": [geometry_filter, date_range_filter, cloud_cover_filter]
}



item_type = "PSScene4Band"

# API request object
search_request = {
  "item_types": [item_type], 
  "filter": combined_filter
}


# fire off the POST request
# search_result = "\\"
    
search_result = requests.post('https://api.planet.com/data/v1/quick-search',
                auth=HTTPBasicAuth(api_key, ''),
                json=search_request)



#%% Analysis of data/ selecting specific images


print(json.dumps(search_result.json(), indent=1))

image_ids = [feature['id'] for feature in search_result.json()['features']]
print(image_ids)




