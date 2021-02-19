"Code of 19/02/2021 meant to read and filter AIS Spire data"

#Imports
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import time
# import cartopy.crs as ccrs

filepath = r'C:/Users/Ruben/Documents/Thesis/Data/AIS/AISdata.csv'

start_time = time.time()

data = pd.read_csv(filepath)

print(data.head())

data_head = data.head()

headers = data.columns

# test_data.drop(to_drop, inplace = True, axis = 'columns')


#%% Visualising

# fig = plt.figure(dpi=400)

# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.coastlines()
# ax.set_global()
# ax.scatter(lon,lat, s=4)

# plt.show()
