# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:23:15 2020

@author: Ruben
"""


#%% Initialization and constants

import json
import requests
import time

# Define urls


url = "https://api.planet.com/data/v1"
stats_url = "{}/stats".format(url)
quick_url = "{}/quick-search".format(url)

"""

#%% To do

# Functions to create: function to only select 1km by 1km area around ship location
# Function to set all filters
# Function to give in AIS data
# Function to determine if satelite passes at time
# Find filter to make sure 100% of area is included
# If everything works: give error for high cloud cover

"""
#%% Function definitions

def p(data):
    print(json.dumps(data, indent=2))
    

def geometry(ship_location, box_size = 0.01):
    
    
    
    return geometry_filter

#%% Input ship data (WIP)
    
ship_location = [10,10]






#%% Start request (new)

api_key = '25bafcdc29b94cc488b89c2ca9539a98'

session = requests.Session()

session.auth = (api_key, "")

res = session.get(url)

#Check if connection is correct

if res.status_code != 200:
    raise Exception('No correct server response')



# Select item types (satelite selection)



#%% Test filter
item_types = ["PSScene4Band"]

date_filter = {
    "type": "DateRangeFilter", # Type of filter -> Date Range
    "field_name": "acquired", # The field to filter on: "acquired" -> Date on which the "image was taken"
    "config": {
        "gte": "2013-01-01T00:00:00.000Z",
        "lte": "2016-01-01T00:00:00.000Z"# "gte" -> Greater than or equal to
    }
}

cloud_filter = {
  "type": "RangeFilter",
  "field_name": "cloud_cover",
  "config": {
    "lt": 0.2,
  }
}

permission_filter = {
  "type": "PermissionFilter",
  "config": ["assets.analytic:download"]
}

geometry_filter = {
  "type": "GeometryFilter",
  "field_name": "geometry",
  "config": {
    "type": "Polygon",
    "coordinates": [
      [
        [
          -120.27282714843749,
          38.348118547988065
        ],
        [
          -120.27282714843749,
          38.74337300148126
        ],
        [
          -119.761962890625,
          38.74337300148126
        ],
        [
          -119.761962890625,
          38.348118547988065
        ],
        [
          -120.27282714843749,
          38.348118547988065
        ]
      ]
    ]
  }
}

total_filter = {
    "type": "AndFilter",
    "config": [date_filter, permission_filter, geometry_filter, cloud_filter]}

#%% Test request (quick search)

request = {
    "item_types" : item_types,
    "filter" : total_filter
    }

response = session.post(quick_url, json=request)

geojson = response.json()

# p(geojson)

response_features = geojson["features"]

# Print amount of images
print(f'{len(response_features)} results found')

## Print list of ID's
# for f in response_features:
#     p(f["id"])


# Print first feature
# p(response_features[0])

# Choosing one of features to display
feature = response_features[3]
# p(feature)

assets_url = feature["_links"]["assets"]

# print(assets_url)

res = session.get(assets_url)

assets = res.json()

print(assets.keys())

udm = assets['udm']


#%% Activating udm asset

activation_url = udm["_links"]["activate"]

res = session.get(activation_url)

p(res.status_code)

#%% Activation check loop

while True:
    if str(res.status_code) == '202':
        time.sleep(500)
    else:
        break
        
#%% Downloading asset

location_url = udm["location"]  

def pl_download(url, filename=None):
    
    # Send a GET request to the provided location url, using your API Key for authentication
    res = requests.get(url, stream=True, auth=(api_key, ""))
    # If no filename argument is given
    if not filename:
        # Construct a filename from the API response
        if "content-disposition" in res.headers:
            filename = res.headers["content-disposition"].split("filename=")[-1].strip("'\"")
        # Construct a filename from the location url
        else:
            filename = url.split("=")[1][:10]
    # Save the file
    with open('C://Users/Ruben/Documents/Thesis/Data/' + filename, "wb") as f:
        for chunk in res.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()

    return filename    
    
pl_download(location_url)    
    

#%% Request to stats
"""
request = {
    "item_types": item_types,
    "interval": "year",
    "filter": date_filter
    }

#Send request

res = session.post(stats_url, json = request)

p(res.json())

"""

#%% Determining area and filters
"""
#https://developers.planet.com/docs/apis/data/searches-filtering/
# Trying an area at Galapagos Isles


# Area in lat,lon (created via geojson.io) 
geojson_geometry = {
  "type": "Polygon",
  "coordinates": [
    [ 
      [
              -90.31173706054688,
              -0.749666757708103
            ],
            [
              -90.30392646789551,
              -0.749666757708103
            ],
            [
              -90.30392646789551,
              -0.7431870906303241
            ],
            [
              -90.31173706054688,
              -0.7431870906303241
            ],
            [
              -90.31173706054688,
              -0.749666757708103
            ]
         
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

area_coverage_filter = {
    "type": "RangeFilter",
    "field_name": "visible_percent",
    "config":{
        "gte": 50
        }
}

# combine our geo, date, cloud filters
combined_filter = {
  "type": "AndFilter",
   # "config": [geometry_filter, date_range_filter, cloud_cover_filter,
   #            area_coverage_filter]
   "config": [geometry_filter, date_range_filter, cloud_cover_filter]
}



item_type = "PSScene4Band"

# API request object
search_request = {
  "item_types": [item_type], 
  "filter": combined_filter
}

"""





