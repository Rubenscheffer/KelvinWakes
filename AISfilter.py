"Code of 19/02/2021 meant to read and filter AIS Spire data"

#%% Imports
# import numpy as np
# import pickle
import pandas as pd
import time
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs

start_time = time.time()


#%% Loading File

filepath = r'C:/Users/Ruben/Documents/Thesis/Data/AIS/AISdata.csv'

data = pd.read_csv(filepath)
headers = data.columns
data_head = data.head()

#%% Filtering File

columns_to_drop = ['created_at', 'eta', 'destination', 'status', 'maneuver']
data.drop(columns_to_drop, inplace=True, axis='columns')

# Joining two measurement sets together

#%% After filtering

headers = data.columns
data_head = data.head()


#%% Visualising

# fig = plt.figure(dpi=400)

# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.coastlines()
# ax.set_global()
# ax.scatter(lon,lat, s=4)

# plt.show()

#%% Final Message

end_time = time.time()
execution_time = end_time - start_time
print(f'Done in {execution_time} seconds')
